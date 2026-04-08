"""
bot_service.py
══════════════════════════════════════════════════════════════════════════════
Telegram bot AI pipeline — three-stage orchestrator:

  Stage 1 — OCR via Claude Vision
    Sends the raw image bytes to Claude claude-opus-4-6 (multimodal) and asks it
    to extract all Tử Vi / horoscope text visible in the image.

  Stage 2 — RAG Retrieval
    Uses the extracted text as a similarity-search query against the MongoDB
    Atlas Vector Store (populated by rag_service.py / POST /api/ingest).
    Returns the top-k most relevant knowledge chunks.

  Stage 3 — Answer Generation via Claude
    Builds a rich prompt from the knowledge context + OCR text, then calls
    Claude claude-opus-4-6 with adaptive thinking to produce a detailed Vietnamese
    Tử Vi interpretation.

Public API:
  analyze_horoscope_image(image_bytes: bytes) -> str
    — Full pipeline. Returns the final answer string to be sent to the user.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from rag_service import get_vector_store

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL: str = "claude-opus-4-6"
RAG_TOP_K: int = 5          # number of knowledge chunks to retrieve
MAX_TOKENS: int = 4096      # max output tokens for the final answer

# ── Anthropic client (module-level singleton) ──────────────────────────────────
_anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── System prompt for the answer generation stage ─────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 1 — OCR: extract Tử Vi text from image using Claude Vision
# ══════════════════════════════════════════════════════════════════════════════

def _ocr_image_with_claude(image_bytes: bytes) -> str:
    """
    Send the image to Claude claude-opus-4-6 Vision and extract all horoscope /
    Tử Vi text visible in it.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG, PNG, WebP, GIF).

    Returns:
        A string containing all extracted text. May be empty if Claude could
        not detect any meaningful content.
    """
    logger.info("Stage 1 — OCR: sending image to Claude Vision (%d bytes)", len(image_bytes))

    # Encode to base64 — Claude Vision requires base64 for inline images.
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    ocr_prompt = (
        "Hãy trích xuất toàn bộ văn bản liên quan đến Tử Vi, lá số, các sao, "
        "cung mệnh, can chi, và mọi thông tin chiêm tinh khác có trong hình ảnh này. "
        "Trả về văn bản thuần túy, giữ nguyên cấu trúc bảng/cột nếu có. "
        "Nếu không tìm thấy nội dung Tử Vi, hãy mô tả những gì bạn thấy trong ảnh."
    )

    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            # Use image/jpeg as the safe default; Telegram always
                            # provides JPEG for photos regardless of original format.
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": ocr_prompt,
                    },
                ],
            }
        ],
    )

    # Extract the text block from Claude's response.
    extracted_text: str = ""
    for block in response.content:
        if block.type == "text":
            extracted_text += block.text

    logger.info("Stage 1 — OCR complete: extracted %d characters", len(extracted_text))
    return extracted_text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 2 — RAG Retrieval: query the vector store with the OCR text
# ══════════════════════════════════════════════════════════════════════════════

def _retrieve_context(query_text: str, top_k: int = RAG_TOP_K) -> list[str]:
    """
    Perform similarity search in MongoDB Atlas Vector Search using the
    extracted OCR text as the query.

    Args:
        query_text: The text extracted from the horoscope image.
        top_k:      Maximum number of knowledge chunks to retrieve.

    Returns:
        A list of page_content strings from the most relevant Documents.
        Returns an empty list if the query is empty or retrieval fails.
    """
    if not query_text:
        logger.warning("Stage 2 — RAG: empty query text, skipping retrieval")
        return []

    logger.info("Stage 2 — RAG: querying vector store (top_k=%d)", top_k)

    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        docs = retriever.get_relevant_documents(query_text)
        context_chunks = [doc.page_content for doc in docs]
        logger.info(
            "Stage 2 — RAG: retrieved %d chunk(s) from knowledge base",
            len(context_chunks),
        )
        return context_chunks

    except Exception:
        # Retrieval is best-effort; the bot can still answer from OCR alone.
        logger.exception("Stage 2 — RAG: retrieval failed, proceeding without context")
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  Stage 3 — Answer Generation: synthesise OCR + context into an interpretation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_answer(raw_text: str, context_chunks: list[str]) -> str:
    """
    Call Claude claude-opus-4-6 with adaptive thinking to generate a detailed Tử Vi
    interpretation in Vietnamese.

    Args:
        raw_text:       OCR text extracted from the horoscope image.
        context_chunks: Relevant knowledge passages retrieved from the vector DB.

    Returns:
        A Vietnamese-language interpretation string.
    """
    logger.info(
        "Stage 3 — Generate: calling Claude with %d context chunk(s)",
        len(context_chunks),
    )

    # ── Build the user message ─────────────────────────────────────────────
    context_section: str = ""
    if context_chunks:
        formatted_chunks = "\n\n---\n\n".join(
            f"[Tài liệu tham khảo {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
        context_section = (
            "## Kiến thức tham khảo từ cơ sở dữ liệu Tử Vi\n\n"
            f"{formatted_chunks}\n\n"
        )

    horoscope_section: str = (
        "## Thông tin từ lá số (trích xuất từ hình ảnh)\n\n"
        f"{raw_text if raw_text else '[Không trích xuất được nội dung từ hình ảnh]'}"
    )

    user_message: str = (
        f"{context_section}"
        f"{horoscope_section}\n\n"
        "---\n\n"
        "Dựa trên thông tin lá số trên và kiến thức tham khảo, "
        "hãy phân tích và luận giải lá số Tử Vi này một cách chi tiết."
    )

    # ── Claude call with adaptive thinking ────────────────────────────────
    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=_ASTROLOGER_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": user_message,
            }
        ],
    )

    # Collect only text blocks (skip thinking blocks — those are internal).
    answer: str = ""
    for block in response.content:
        if block.type == "text":
            answer += block.text

    logger.info("Stage 3 — Generate: answer produced (%d chars)", len(answer))
    return answer.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Public Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def analyze_horoscope_image(image_bytes: bytes) -> str:
    """
    Full three-stage pipeline: OCR → RAG Retrieval → Answer Generation.

    Args:
        image_bytes: Raw bytes of the horoscope image downloaded from Telegram.

    Returns:
        A Vietnamese Tử Vi interpretation string ready to be sent back to the
        Telegram user. Never raises — falls back to an error message string on
        unexpected failures so the bot always replies.
    """
    try:
        # Stage 1 — Extract text from image via Claude Vision.
        raw_text: str = _ocr_image_with_claude(image_bytes)

        # Stage 2 — Retrieve relevant knowledge from MongoDB Atlas.
        context_chunks: list[str] = _retrieve_context(raw_text)

        # Stage 3 — Generate final answer with Claude.
        answer: str = _generate_answer(raw_text, context_chunks)

        return answer if answer else (
            "⚠️ Xin lỗi, tôi không thể tạo ra phân tích cho lá số này. "
            "Vui lòng thử lại với hình ảnh rõ hơn."
        )

    except anthropic.AuthenticationError:
        logger.error("Anthropic API key is invalid")
        return "❌ Lỗi xác thực API. Vui lòng liên hệ quản trị viên."

    except anthropic.RateLimitError:
        logger.warning("Anthropic rate limit hit")
        return (
            "⏳ Hệ thống đang bận. Vui lòng thử lại sau vài giây."
        )

    except Exception:
        logger.exception("Unexpected error in analyze_horoscope_image")
        return (
            "❌ Đã xảy ra lỗi khi phân tích hình ảnh. "
            "Vui lòng thử lại hoặc liên hệ quản trị viên."
        )
