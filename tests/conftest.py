"""
Shared pytest fixtures and mock implementations.

All mocks implement the abstract interfaces from app.core.interfaces
so they can be injected into the RAGPipeline and tested without any
external dependencies (no database, no API calls).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Environment: provide minimal required env vars before any app module loads
# ---------------------------------------------------------------------------
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-dummy-for-tests")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("GOOGLE_API_KEY", "dummy-google-key")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "1234567890:dummy-token")

from app.core.interfaces import (  # noqa: E402
    EmbeddingService,
    LLMService,
    TuViEnginePort,
    VectorStoreRepository,
)
from app.domain.models import BirthData, ChatMessage, Chunk, Gender, MessageRole  # noqa: E402


# =============================================================================
#  In-Memory Mock Implementations
# =============================================================================


class InMemoryVectorStore(VectorStoreRepository):
    """Thread-safe, in-memory vector store for unit tests."""

    def __init__(self) -> None:
        self._documents: list[Chunk] = []
        self._messages: list[ChatMessage] = []
        self._users: dict[str, dict[str, Any]] = {}

    # ── Knowledge base ────────────────────────────────────────────────────

    def add_documents(self, chunks: list[Chunk]) -> int:
        self._documents.extend(chunks)
        return len(chunks)

    def similarity_search(self, query: str, *, top_k: int = 5) -> list[str]:
        return [c.content for c in self._documents[:top_k]]

    # ── Chat history ──────────────────────────────────────────────────────

    def save_chat_message(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def get_recent_messages(self, user_id: str, *, limit: int = 5) -> list[ChatMessage]:
        user_msgs = [m for m in self._messages if m.user_id == user_id]
        return user_msgs[-limit:][::-1]

    def vector_search_messages(
        self,
        query_embedding: list[float],
        user_id: str,
        *,
        top_k: int = 5,
    ) -> list[ChatMessage]:
        user_msgs = [m for m in self._messages if m.user_id == user_id]
        return user_msgs[:top_k]

    # ── User persistence ──────────────────────────────────────────────────

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        return self._users.get(user_id)

    def upsert_user(self, user_id: str, data: dict[str, Any]) -> None:
        if user_id not in self._users:
            self._users[user_id] = {}
        self._users[user_id].update(data)


class FakeLLMService(LLMService):
    """Deterministic LLM stub that returns preset responses."""

    def __init__(self, response: str = "Fake LLM answer") -> None:
        self.response = response
        self.generate_calls: list[dict[str, Any]] = []
        self.extract_calls: list[dict[str, Any]] = []
        self._extract_response: dict[str, Any] | None = None

    def set_extract_response(self, data: dict[str, Any] | None) -> None:
        """Pre-configure what extract_structured should return."""
        self._extract_response = data

    def generate(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        self.generate_calls.append(
            {"system_prompt": system_prompt, "user_message": user_message, "max_tokens": max_tokens}
        )
        return self.response

    def generate_with_cache(
        self,
        *,
        system_prompt: str,
        chart_json: dict,
        conversation_context: str,
        query: str,
        max_tokens: int = 16000,
    ) -> str:
        # Reconstruct a user_message compatible with existing test assertions.
        import json
        parts = []
        if chart_json:
            parts.append(
                "## La so Tu Vi cua nguoi dung\n\n"
                f"```json\n{json.dumps(chart_json, ensure_ascii=False, indent=2)}\n```"
            )
        if conversation_context:
            parts.append(conversation_context)
        parts.append(f"## Cau hoi hien tai\n\n{query}")
        user_message = "\n\n---\n\n".join(parts)
        self.generate_calls.append(
            {"system_prompt": system_prompt, "user_message": user_message, "max_tokens": max_tokens}
        )
        return self.response

    def extract_structured(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> dict[str, Any] | None:
        self.extract_calls.append(
            {"system_prompt": system_prompt, "user_message": user_message, "max_tokens": max_tokens}
        )
        return self._extract_response


class FakeEmbeddingService(EmbeddingService):
    """Returns deterministic zero-vectors for testing."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.embed_text_calls: list[str] = []
        self.embed_documents_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        self.embed_text_calls.append(text)
        return [0.0] * self.dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(texts)
        return [[0.0] * self.dim for _ in texts]


class FakeTuViEngine(TuViEnginePort):
    """Returns a hard-coded chart dict for testing."""

    def __init__(self) -> None:
        self.call_count = 0
        self.last_birth_data: BirthData | None = None

    def generate_chart(self, birth_data: BirthData) -> dict[str, Any]:
        self.call_count += 1
        self.last_birth_data = birth_data
        return {
            "ho_ten": birth_data.name,
            "gioi_tinh": birth_data.gender.value,
            "ngay_sinh_duong": birth_data.solar_dob,
            "menh_cung": "Thien Di",
        }


# =============================================================================
#  Fixtures
# =============================================================================


@pytest.fixture()
def vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture()
def llm() -> FakeLLMService:
    return FakeLLMService()


@pytest.fixture()
def embedding() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture()
def tuvi_engine() -> FakeTuViEngine:
    return FakeTuViEngine()


@pytest.fixture()
def sample_birth_data() -> BirthData:
    return BirthData(
        name="Nguyen Van A",
        gender=Gender.MALE,
        solar_dob="1990-05-15",
        birth_hour=0,
    )


@pytest.fixture()
def sample_user_id() -> str:
    return "user_12345"
