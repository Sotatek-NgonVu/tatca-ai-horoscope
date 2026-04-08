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
    "Toi co the phan tich la so Tu Vi tu hinh anh.\n\n"
    "*Cach su dung:*\n"
    "Gui cho toi mot hinh anh chua la so Tu Vi — "
    "toi se doc va luan giai no cho ban.\n\n"
    "Luu y: Qua trinh phan tich co the mat 15-30 giay."
)

THINKING_MESSAGE = (
    "Dang phan tich la so cua ban...\n"
    "_(Qua trinh nay co the mat 15-30 giay)_"
)

TEXT_GUIDE_MESSAGE = (
    "Vui long gui *hinh anh* la so Tu Vi cua ban.\n"
    "Toi se doc va luan giai la so cho ban."
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


async def handle_text_message(
    client: TelegramClient,
    chat_id: int,
    message_id: int,
) -> None:
    """Guide the user to send an image instead of text."""
    await client.send_message(
        chat_id=chat_id,
        text=TEXT_GUIDE_MESSAGE,
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
    Background task: download image -> AI pipeline -> reply to user.

    This runs after the webhook has already returned 200 to Telegram.
    """
    try:
        # Download the image
        image_bytes = await client.download_file(file_id)

        # Run the sync AI pipeline in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            pipeline.analyze_image,
            image_bytes,
        )

        # Reply with the answer (split if > 4096 chars)
        answer = result.answer
        if len(answer) <= 4096:
            await client.send_message(
                chat_id=chat_id,
                text=answer,
                reply_to_message_id=message_id,
            )
        else:
            chunks = split_long_message(answer)
            for i, chunk in enumerate(chunks):
                reply_id = message_id if i == 0 else None
                await client.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_to_message_id=reply_id,
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
