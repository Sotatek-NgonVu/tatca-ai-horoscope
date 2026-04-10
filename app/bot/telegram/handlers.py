"""
Telegram message handlers.

Pure functions that handle specific message types.
Each handler receives a TelegramClient and RAGPipeline via parameters
(no global state).
"""

from __future__ import annotations

import asyncio
import logging

from app.bot.telegram.client import TelegramClient
from app.domain.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ── Bot messages ─────────────────────────────────────────────────────────────

WELCOME_MESSAGE = (
    "*Chao mung ban den voi Bot Tu Vi AI!*\n\n"
    "Toi la chuyen gia Tu Vi Dong Phuong, san sang luan giai la so cho ban.\n\n"
    "*Cach su dung:*\n"
    "- Gui tin nhan bat ky de tro chuyen.\n"
    "- Toi se yeu cau thong tin ngay sinh cua ban (neu chua co) de lap la so Tu Vi.\n"
    "- Sau do, hay hoi bat cu dieu gi ve la so cua ban!\n\n"
    "Bat dau nao — hay gui cho toi mot cau hoi!"
)

THINKING_MESSAGE = (
    "Dang phan tich la so cua ban...\n"
    "_(Qua trinh nay co the mat 15-30 giay)_"
)

TEXT_GUIDE_MESSAGE = (
    "Vui long gui tin nhan de tro chuyen voi Tu Vi AI.\n"
    "Toi se giup ban luan giai la so Tu Vi!"
)


# ── Utility ──────────────────────────────────────────────────────────────────

def split_long_message(text: str, max_len: int = 4000) -> list[str]:
    """
    Split a long string into chunks of at most `max_len` characters,
    breaking at newline or space boundaries where possible.
    """
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # Try to break at a newline.
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos == -1:
            # Fall back to last space.
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos == -1:
            # Hard cut.
            split_pos = max_len
        parts.append(text[:split_pos].rstrip())
        text = text[split_pos:].lstrip()

    return parts


# ── Handlers ─────────────────────────────────────────────────────────────────

async def handle_start_command(
    client: TelegramClient,
    chat_id: int,
    message_id: int,
) -> None:
    """Handle the /start command."""
    await client.send_message(
        chat_id=chat_id,
        text=WELCOME_MESSAGE,
        reply_to_message_id=message_id,
    )


async def handle_photo_message(
    client: TelegramClient,
    pipeline: RAGPipeline,
    chat_id: int,
    file_id: str,
    message_id: int,
) -> None:
    """
    Background task: download image -> OCR -> treat as text query.

    In v3 the primary flow is text-based, so photo messages are
    OCR'd and then fed into the regular chat pipeline.
    """
    try:
        # Download the image
        image_bytes = await client.download_file(file_id)

        # OCR the image (if OCR service is available in the pipeline)
        # For now, acknowledge the photo and ask the user to type instead
        await client.send_message(
            chat_id=chat_id,
            text=(
                "Toi da nhan duoc hinh anh cua ban. "
                "Hien tai toi ho tro tot nhat qua tin nhan van ban.\n\n"
                "Hay gui cau hoi cua ban bang tin nhan nhe!"
            ),
            reply_to_message_id=message_id,
        )

    except Exception:
        logger.exception(
            "Error in photo processing for chat_id=%s", chat_id
        )
        await client.send_message(
            chat_id=chat_id,
            text="Da xay ra loi khi xu ly hinh anh. Vui long thu lai.",
            reply_to_message_id=message_id,
        )
