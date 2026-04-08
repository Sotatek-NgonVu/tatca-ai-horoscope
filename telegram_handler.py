"""
telegram_handler.py
══════════════════════════════════════════════════════════════════════════════
FastAPI router that handles incoming Telegram Webhook updates.

Telegram sends a POST request to /telegram/webhook for every event
(message, photo, command, etc.).  This handler:

  1. Parses the incoming Telegram Update JSON.
  2. Routes to the appropriate handler:
       • Photo message  → download image → AI pipeline → reply
       • /start command → send welcome message
       • Text message   → guide user to send a photo
       • Anything else  → silently ignore (return 200 quickly)
  3. ALWAYS returns HTTP 200 to Telegram within a few seconds, even if the
     AI pipeline is still running (we fire it as a background task so
     Telegram does not retry the webhook delivery).

Mounting:
  In main.py:  app.include_router(telegram_router)
  The webhook URL registered with Telegram: <WEBHOOK_BASE_URL>/telegram/webhook
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse

from bot_service import analyze_horoscope_image

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# ── Telegram Bot API ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_API_BASE: str = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
TELEGRAM_FILE_BASE: str = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"

# ── Router ────────────────────────────────────────────────────────────────────
telegram_router = APIRouter(prefix="/telegram", tags=["Telegram Bot"])

# ── Welcome / help text ───────────────────────────────────────────────────────
_WELCOME_MESSAGE = (
    "🌟 *Chào mừng bạn đến với Bot Tử Vi AI!*\n\n"
    "Tôi có thể phân tích lá số Tử Vi từ hình ảnh.\n\n"
    "📸 *Cách sử dụng:*\n"
    "Gửi cho tôi một hình ảnh chứa lá số Tử Vi — "
    "tôi sẽ đọc và luận giải nó cho bạn.\n\n"
    "⚡ Lưu ý: Quá trình phân tích có thể mất 15–30 giây."
)

_THINKING_MESSAGE = (
    "🔍 Đang phân tích lá số của bạn...\n"
    "_(Quá trình này có thể mất 15–30 giây)_"
)

_TEXT_GUIDE_MESSAGE = (
    "📸 Vui lòng gửi *hình ảnh* lá số Tử Vi của bạn.\n"
    "Tôi sẽ đọc và luận giải lá số cho bạn."
)


# ══════════════════════════════════════════════════════════════════════════════
#  Telegram HTTP Helpers
# ══════════════════════════════════════════════════════════════════════════════

async def _telegram_send_message(
    chat_id: int | str,
    text: str,
    parse_mode: str = "Markdown",
    reply_to_message_id: int | None = None,
) -> None:
    """
    Send a text message to a Telegram chat via Bot API.

    Args:
        chat_id:              Target chat / user ID.
        text:                 Message body (supports Markdown by default).
        parse_mode:           Telegram parse mode ('Markdown' or 'HTML').
        reply_to_message_id:  Optional message ID to reply to.
    """
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
    }
    if reply_to_message_id:
        payload["reply_to_message_id"] = reply_to_message_id

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{TELEGRAM_API_BASE}/sendMessage", json=payload)
        if not resp.is_success:
            logger.error(
                "sendMessage failed: status=%d body=%s", resp.status_code, resp.text
            )


async def _telegram_download_photo(file_id: str) -> bytes:
    """
    Download a Telegram photo by its file_id and return the raw bytes.

    Telegram's largest photo variant is always the last element of the
    photo array (highest resolution).

    Args:
        file_id: Telegram file_id string for the photo.

    Returns:
        Raw image bytes.

    Raises:
        httpx.HTTPStatusError: If the Telegram API or CDN returns an error.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1 — Resolve file_id → file_path on Telegram CDN.
        get_file_resp = await client.get(
            f"{TELEGRAM_API_BASE}/getFile",
            params={"file_id": file_id},
        )
        get_file_resp.raise_for_status()
        file_path: str = get_file_resp.json()["result"]["file_path"]

        # Step 2 — Download the actual bytes.
        download_url = f"{TELEGRAM_FILE_BASE}/{file_path}"
        download_resp = await client.get(download_url)
        download_resp.raise_for_status()

        logger.info(
            "Downloaded photo (file_id=%s): %d bytes", file_id, len(download_resp.content)
        )
        return download_resp.content


# ══════════════════════════════════════════════════════════════════════════════
#  Background Task — Photo Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

async def _process_photo_and_reply(
    chat_id: int | str,
    file_id: str,
    original_message_id: int,
) -> None:
    """
    Background task: download image → AI pipeline → reply to user.

    Runs after the webhook handler has already returned 200 to Telegram,
    so there's no risk of Telegram timing out and retrying.

    Args:
        chat_id:             Chat to reply in.
        file_id:             Telegram file_id of the largest photo variant.
        original_message_id: Message to reply to (threads the response).
    """
    try:
        # ── Download ──────────────────────────────────────────────────────────
        image_bytes: bytes = await _telegram_download_photo(file_id)

        # ── AI Pipeline (sync → run in thread pool via asyncio) ────────────
        # analyze_horoscope_image is sync (uses blocking Anthropic SDK calls).
        # Running it in the default executor keeps the event loop unblocked.
        import asyncio
        loop = asyncio.get_event_loop()
        answer: str = await loop.run_in_executor(
            None,                         # default ThreadPoolExecutor
            analyze_horoscope_image,
            image_bytes,
        )

        # ── Reply ─────────────────────────────────────────────────────────────
        # Telegram message limit is 4096 chars; split if needed.
        if len(answer) <= 4096:
            await _telegram_send_message(
                chat_id=chat_id,
                text=answer,
                reply_to_message_id=original_message_id,
            )
        else:
            # Send in chunks of 4000 chars, preserving word boundaries.
            chunks = _split_long_message(answer, max_len=4000)
            for i, chunk in enumerate(chunks):
                reply_id = original_message_id if i == 0 else None
                await _telegram_send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_to_message_id=reply_id,
                )

    except Exception:
        logger.exception(
            "Error in background photo processing for chat_id=%s", chat_id
        )
        await _telegram_send_message(
            chat_id=chat_id,
            text="❌ Đã xảy ra lỗi khi xử lý hình ảnh. Vui lòng thử lại.",
            reply_to_message_id=original_message_id,
        )


def _split_long_message(text: str, max_len: int = 4000) -> list[str]:
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


# ══════════════════════════════════════════════════════════════════════════════
#  Webhook Endpoint
# ══════════════════════════════════════════════════════════════════════════════

@telegram_router.post(
    "/webhook",
    summary="Telegram Webhook Receiver",
    include_in_schema=False,   # Hide from public /docs — security by obscurity.
)
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> JSONResponse:
    """
    Receive and dispatch Telegram Update objects.

    Telegram requires this endpoint to respond with HTTP 200 within 5 seconds.
    Heavy work (AI pipeline) is offloaded to `background_tasks` so we can
    acknowledge the webhook immediately.

    Supported update types:
      - Photo message  → enqueue background AI pipeline
      - /start command → reply with welcome message
      - Text message   → guide user to send a photo
    """
    try:
        update: dict[str, Any] = await request.json()
    except Exception:
        logger.warning("Failed to parse webhook JSON body")
        return JSONResponse({"ok": True})

    logger.debug("Telegram update received: %s", update.get("update_id"))

    # ── Extract message ────────────────────────────────────────────────────
    message: dict[str, Any] | None = update.get("message") or update.get("edited_message")
    if not message:
        # Ignore non-message updates (channel posts, inline queries, etc.).
        return JSONResponse({"ok": True})

    chat_id: int = message["chat"]["id"]
    message_id: int = message["message_id"]

    # ── Route: Photo ──────────────────────────────────────────────────────
    if "photo" in message:
        # Telegram provides multiple sizes; take the last (highest resolution).
        best_photo = message["photo"][-1]
        file_id: str = best_photo["file_id"]

        logger.info(
            "Photo received: chat_id=%s, file_id=%s, size=%dx%d",
            chat_id,
            file_id,
            best_photo.get("width", 0),
            best_photo.get("height", 0),
        )

        # Send an immediate "thinking" acknowledgement.
        await _telegram_send_message(
            chat_id=chat_id,
            text=_THINKING_MESSAGE,
            reply_to_message_id=message_id,
        )

        # Dispatch the heavy work to a background task.
        background_tasks.add_task(
            _process_photo_and_reply,
            chat_id=chat_id,
            file_id=file_id,
            original_message_id=message_id,
        )
        return JSONResponse({"ok": True})

    # ── Route: Text / Commands ─────────────────────────────────────────────
    text: str = (message.get("text") or "").strip()

    if text.startswith("/start"):
        await _telegram_send_message(
            chat_id=chat_id,
            text=_WELCOME_MESSAGE,
            reply_to_message_id=message_id,
        )
        return JSONResponse({"ok": True})

    if text:
        await _telegram_send_message(
            chat_id=chat_id,
            text=_TEXT_GUIDE_MESSAGE,
            reply_to_message_id=message_id,
        )
        return JSONResponse({"ok": True})

    # ── Anything else — silently acknowledge ──────────────────────────────
    return JSONResponse({"ok": True})
