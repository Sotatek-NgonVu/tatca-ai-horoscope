"""
Telegram webhook router.

FastAPI router that receives Telegram Update objects and dispatches
them to the appropriate handler.  Returns HTTP 200 immediately for
all webhook deliveries (Telegram requirement --- must respond within
5 seconds).

Heavy work (RAG pipeline, LLM calls) is offloaded to ``BackgroundTasks``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse

from app.bot.telegram.client import TelegramClient
from app.bot.telegram.handlers import (
    THINKING_MESSAGE,
    handle_photo_message,
    handle_start_command,
    split_long_message,
)
from app.domain.pipeline import RAGPipeline
from app.infrastructure.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)

telegram_router = APIRouter(prefix="/telegram", tags=["Telegram Bot"])


# ── Module-level client reference (set during app startup) ───────────────────
_telegram_client: TelegramClient | None = None


def set_telegram_client(client: TelegramClient) -> None:
    """Set the module-level Telegram client (called during startup)."""
    global _telegram_client  # noqa: PLW0603
    _telegram_client = client


def get_telegram_client() -> TelegramClient:
    """Return the Telegram client for dependency injection."""
    if _telegram_client is None:
        raise RuntimeError("TelegramClient not initialized")
    return _telegram_client


# ═════════════════════════════════════════════════════════════════════════════
#  Background task: process a text message through the RAG pipeline
# ═════════════════════════════════════════════════════════════════════════════


async def _process_text_message(
    client: TelegramClient,
    pipeline: RAGPipeline,
    chat_id: int,
    text: str,
    message_id: int,
) -> None:
    """
    Background task: run the user's text query through the RAG pipeline
    and send the result back via Telegram.

    The pipeline is synchronous (uses pymongo + sentence-transformers),
    so we run it in a thread pool to avoid blocking the event loop.
    """
    try:
        user_id = str(chat_id)

        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            pipeline.chat,
            user_id,
            text,
        )

        # Send the answer (split if > 4096 chars for Telegram's limit)
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
            "Error processing text message for chat_id=%s", chat_id
        )
        await client.send_message(
            chat_id=chat_id,
            text="Da xay ra loi khi xu ly tin nhan. Vui long thu lai.",
            reply_to_message_id=message_id,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  Webhook endpoint
# ═════════════════════════════════════════════════════════════════════════════


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

    # ── Route: Photo ─────────────────────────────────────────────────────
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

    # ── Route: Text / Commands ───────────────────────────────────────────
    text: str = (message.get("text") or "").strip()

    if text.startswith("/start"):
        await handle_start_command(client, chat_id, message_id)
        return JSONResponse({"ok": True})

    if text:
        # Send "Dang xu ly..." acknowledgement immediately
        await client.send_message(
            chat_id=chat_id,
            text="Dang xu ly...",
            reply_to_message_id=message_id,
        )

        # Offload RAG pipeline to background task
        background_tasks.add_task(
            _process_text_message,
            client=client,
            pipeline=pipeline,
            chat_id=chat_id,
            text=text,
            message_id=message_id,
        )
        return JSONResponse({"ok": True})

    # ── Anything else --- silently acknowledge ───────────────────────────
    return JSONResponse({"ok": True})
