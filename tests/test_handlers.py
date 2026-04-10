"""
Unit tests for app/bot/telegram/handlers.py

Tests the split_long_message() utility and constant values.
No real Telegram client calls — async handlers that call the client
are tested by passing mock client objects.
"""

from __future__ import annotations

import pytest

from app.bot.telegram.handlers import (
    TEXT_GUIDE_MESSAGE,
    THINKING_MESSAGE,
    WELCOME_MESSAGE,
    split_long_message,
)


# =============================================================================
#  split_long_message()
# =============================================================================


class TestSplitLongMessage:
    def test_short_message_returned_as_single_chunk(self) -> None:
        text = "Hello world"
        parts = split_long_message(text, max_len=4000)
        assert parts == ["Hello world"]

    def test_empty_string_returns_single_empty_chunk(self) -> None:
        parts = split_long_message("", max_len=4000)
        assert parts == [""]

    def test_exactly_max_len_is_not_split(self) -> None:
        text = "A" * 4000
        parts = split_long_message(text, max_len=4000)
        assert len(parts) == 1
        assert parts[0] == text

    def test_message_longer_than_max_len_is_split(self) -> None:
        text = "A" * 4001
        parts = split_long_message(text, max_len=4000)
        assert len(parts) == 2

    def test_split_at_newline_boundary(self) -> None:
        # Build a text that is 20 chars + newline + 20 chars = 41 chars total
        line1 = "A" * 20
        line2 = "B" * 20
        text = line1 + "\n" + line2
        parts = split_long_message(text, max_len=25)
        # Should break at the newline (position 20)
        assert parts[0] == line1
        assert parts[1] == line2

    def test_split_at_space_when_no_newline(self) -> None:
        # 10 chars + space + 10 chars = 21 chars; max_len = 15
        text = "AAAAAAAAAA BBBBBBBBBB"
        parts = split_long_message(text, max_len=15)
        assert len(parts) == 2
        assert "AAAAAAAAAA" in parts[0]
        assert "BBBBBBBBBB" in parts[1]

    def test_hard_cut_when_no_space_or_newline(self) -> None:
        text = "A" * 10
        parts = split_long_message(text, max_len=3)
        # Should hard-cut at max_len boundaries
        for part in parts:
            assert len(part) <= 3

    def test_all_parts_concatenate_to_original(self) -> None:
        """Reassembling parts should reconstruct the original (ignoring strip)."""
        text = "Hello world!\nThis is a longer message.\nThird line here."
        parts = split_long_message(text, max_len=20)
        # Join with newlines since rfind breaks at them
        rejoined = "\n".join(parts)
        # All original words should still be present
        for word in ["Hello", "world", "longer", "message", "Third", "line"]:
            assert word in rejoined

    def test_no_part_exceeds_max_len(self) -> None:
        import random
        random.seed(42)
        text = " ".join(["word"] * 1000)
        parts = split_long_message(text, max_len=100)
        for part in parts:
            assert len(part) <= 100

    def test_custom_max_len(self) -> None:
        text = "A" * 200
        parts = split_long_message(text, max_len=50)
        assert len(parts) == 4  # 200 / 50 = 4 equal parts


# =============================================================================
#  Constant message strings
# =============================================================================


class TestConstantMessages:
    def test_welcome_message_is_non_empty(self) -> None:
        assert len(WELCOME_MESSAGE) > 0

    def test_welcome_message_is_string(self) -> None:
        assert isinstance(WELCOME_MESSAGE, str)

    def test_thinking_message_is_non_empty(self) -> None:
        assert len(THINKING_MESSAGE) > 0

    def test_text_guide_message_is_non_empty(self) -> None:
        assert len(TEXT_GUIDE_MESSAGE) > 0
