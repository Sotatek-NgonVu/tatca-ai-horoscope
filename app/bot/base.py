"""
Abstract base for bot platform handlers.

Future platforms (API, Zalo, etc.) implement this interface.
Currently only Telegram is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBotHandler(ABC):
    """
    Abstract bot handler interface.

    Each platform (Telegram, API, etc.) implements this to provide
    a consistent way for the application to interact with users.
    """

    @abstractmethod
    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        """Send a text message to a user/chat."""

    @abstractmethod
    async def download_file(self, file_id: str) -> bytes:
        """Download a file by its platform-specific ID."""
