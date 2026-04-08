"""
Claude Vision OCR service.

Extracts text from images using Claude's multimodal (vision) capabilities.
Specifically tuned for Vietnamese Tử Vi / horoscope chart images.
"""

from __future__ import annotations

import base64
import logging

import anthropic

from app.core.exceptions import LLMAuthenticationError, LLMRateLimitError, OCRError
from app.core.interfaces import OCRService

logger = logging.getLogger(__name__)

# Vietnamese OCR prompt for Tử Vi content
_OCR_PROMPT = (
    "Hãy trích xuất toàn bộ văn bản liên quan đến Tử Vi, lá số, các sao, "
    "cung mệnh, can chi, và mọi thông tin chiêm tinh khác có trong hình ảnh này. "
    "Trả về văn bản thuần túy, giữ nguyên cấu trúc bảng/cột nếu có. "
    "Nếu không tìm thấy nội dung Tử Vi, hãy mô tả những gì bạn thấy trong ảnh."
)


class ClaudeOCRService(OCRService):
    """OCR service using Claude Vision for text extraction from images."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def extract_text(
        self, image_bytes: bytes, *, media_type: str = "image/jpeg"
    ) -> str:
        """
        Send the image to Claude Vision and extract all Tử Vi text.

        Args:
            image_bytes: Raw bytes of the image (JPEG, PNG, WebP, GIF).
            media_type: MIME type of the image.

        Returns:
            Extracted text string. May be empty if no content detected.

        Raises:
            OCRError: If the Vision API call fails.
        """
        logger.info(
            "OCR: sending image to Claude Vision (%d bytes, type=%s)",
            len(image_bytes),
            media_type,
        )

        try:
            b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": b64_image,
                                },
                            },
                            {"type": "text", "text": _OCR_PROMPT},
                        ],
                    }
                ],
            )

            extracted = "".join(
                block.text for block in response.content if block.type == "text"
            )

            logger.info("OCR complete: extracted %d characters", len(extracted))
            return extracted.strip()

        except anthropic.AuthenticationError as exc:
            raise OCRError(
                "Invalid Anthropic API key for OCR",
                cause=LLMAuthenticationError("Auth failed", cause=exc),
            ) from exc

        except anthropic.RateLimitError as exc:
            raise OCRError(
                "Rate limit hit during OCR",
                cause=LLMRateLimitError("Rate limit", cause=exc),
            ) from exc

        except Exception as exc:
            raise OCRError(
                f"OCR extraction failed: {exc}", cause=exc
            ) from exc
