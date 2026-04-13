"""
Telegram Bot API HTTP client.

Encapsulates all HTTP communication with the Telegram Bot API.
Implements BaseBotHandler for platform abstraction.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.bot.base import BaseBotHandler

logger = logging.getLogger(__name__)


class TelegramClient(BaseBotHandler):
    """
    HTTP client for the Telegram Bot API.

    Handles sending messages and downloading files via the Bot API.
    A single persistent ``httpx.AsyncClient`` is reused for all requests to
    avoid per-call TCP handshake overhead.
    """

    def __init__(self, bot_token: str) -> None:
        self._bot_token = bot_token
        self._api_base = f"https://api.telegram.org/bot{bot_token}"
        self._file_base = f"https://api.telegram.org/file/bot{bot_token}"
        # Reuse a single AsyncClient — avoids per-call TCP overhead.
        self._http = httpx.AsyncClient(timeout=60.0)

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Call during application shutdown."""
        await self._http.aclose()

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        parse_mode: str = "Markdown",
    ) -> None:
        """Send a text message to a Telegram chat."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_to_message_id:
            payload["reply_to_message_id"] = reply_to_message_id

        resp = await self._http.post(
            f"{self._api_base}/sendMessage", json=payload
        )
        if not resp.is_success:
            logger.error(
                "sendMessage failed: status=%d body=%s",
                resp.status_code,
                resp.text,
            )

    async def send_photo(
        self,
        chat_id: int | str,
        photo_bytes: bytes,
        *,
        caption: str = "",
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send a photo (PNG bytes) to a Telegram chat."""
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        if reply_to_message_id:
            data["reply_to_message_id"] = str(reply_to_message_id)

        files = {"photo": ("chart.png", photo_bytes, "image/png")}

        resp = await self._http.post(
            f"{self._api_base}/sendPhoto", data=data, files=files,
        )
        if not resp.is_success:
            logger.error(
                "sendPhoto failed: status=%d body=%s",
                resp.status_code,
                resp.text,
            )

    async def download_file(self, file_id: str) -> bytes:
        """
        Download a file from Telegram by file_id.

        Steps:
            1. Resolve file_id -> file_path via getFile API
            2. Download the actual bytes from Telegram CDN
        """
        # Resolve file_id -> file_path
        get_file_resp = await self._http.get(
            f"{self._api_base}/getFile",
            params={"file_id": file_id},
        )
        get_file_resp.raise_for_status()
        file_path: str = get_file_resp.json()["result"]["file_path"]

        # Download bytes
        download_url = f"{self._file_base}/{file_path}"
        download_resp = await self._http.get(download_url)
        download_resp.raise_for_status()

        logger.info(
            "Downloaded file (file_id=%s): %d bytes",
            file_id,
            len(download_resp.content),
        )
        return download_resp.content

    async def register_webhook(self, webhook_url: str) -> bool:
        """
        Register a webhook URL with Telegram.

        Returns True if successful, False otherwise.
        """
        logger.info("Registering Telegram webhook: %s", webhook_url)
        try:
            resp = await self._http.post(
                f"{self._api_base}/setWebhook",
                json={
                    "url": webhook_url,
                    "allowed_updates": ["message", "edited_message"],
                    "drop_pending_updates": True,
                },
            )
            result = resp.json()
            if result.get("ok"):
                logger.info("Telegram webhook registered successfully")
                return True
            else:
                logger.error(
                    "Failed to register webhook: %s",
                    result.get("description"),
                )
                return False
        except Exception:
            logger.exception("Exception while registering Telegram webhook")
            return False
