"""
Telegram webhook router.

FastAPI router that receives Telegram Update objects and dispatches
them to the appropriate handler. Returns HTTP 200 immediately for
all webhook deliveries (Telegram requirement).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse

from app.bot.telegram.client import TelegramClient
from app.bot.telegram.handlers import (
    THINKING_MESSAGE,
    handle_photo_message,
    handle_start_command,
    handle_text_message,
)
from app.domain.pipeline import RAGPipeline
from app.infrastructure.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)

telegram_router = APIRouter(prefix="/telegram", tags=["Telegram Bot"])

# ── Module-level client reference (set during app startup) ───────────────────
_telegram_client: TelegramClient | None = None


def set_telegram_client(client: TelegramClient) -> None:
    """Set the module-level Telegram client (called during startup)."""
    global _telegram_client
    _telegram_client = client


def get_telegram_client() -> TelegramClient:
    """Return the Telegram client for dependency injection."""
    if _telegram_client is None:
        raise RuntimeError("TelegramClient not initialized")
    return _telegram_client


@telegram_router.post(
    "/webhook",
    summary="Telegram Webhook Receiver",
    include_in_schema=False,
)
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> JSONResponse:
    """
    Receive and dispatch Telegram Update objects.

    Always returns HTTP 200 within 5 seconds (Telegram requirement).
    Heavy work is offloaded to background tasks.
    """
    client = get_telegram_client()

    try:
        update: dict[str, Any] = await request.json()
    except Exception:
        logger.warning("Failed to parse webhook JSON body")
        return JSONResponse({"ok": True})

    logger.debug("Telegram update received: %s", update.get("update_id"))

    # Extract message
    message: dict[str, Any] | None = (
        update.get("message") or update.get("edited_message")
    )
    if not message:
        return JSONResponse({"ok": True})

    chat_id: int = message["chat"]["id"]
    message_id: int = message["message_id"]

    # ── Route: Photo ──────────────────────────────────────────────────────
    if "photo" in message:
        best_photo = message["photo"][-1]
        file_id: str = best_photo["file_id"]

        logger.info(
            "Photo received: chat_id=%s, file_id=%s, size=%dx%d",
            chat_id,
            file_id,
            best_photo.get("width", 0),
            best_photo.get("height", 0),
        )

        # Send immediate acknowledgement
        await client.send_message(
            chat_id=chat_id,
            text=THINKING_MESSAGE,
            reply_to_message_id=message_id,
        )

        # Dispatch heavy work to background
        background_tasks.add_task(
            handle_photo_message,
            client=client,
            pipeline=pipeline,
            chat_id=chat_id,
            file_id=file_id,
            message_id=message_id,
        )
        return JSONResponse({"ok": True})

    # ── Route: Text / Commands ────────────────────────────────────────────
    text: str = (message.get("text") or "").strip()

    if text.startswith("/start"):
        await handle_start_command(client, chat_id, message_id)
        return JSONResponse({"ok": True})

    if text:
        await handle_text_message(client, chat_id, message_id)
        return JSONResponse({"ok": True})

    # ── Anything else — silently acknowledge ──────────────────────────────
    return JSONResponse({"ok": True})
