"""
Domain models --- pure data structures with no external dependencies.

These are the core objects passed between layers.
All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
#  Enums
# ═════════════════════════════════════════════════════════════════════════════


class Gender(str, Enum):
    """Biological gender, required for Tu Vi chart calculation."""

    MALE = "Nam"
    FEMALE = "Nu"


class MessageRole(str, Enum):
    """Role of a chat message participant."""

    USER = "user"
    ASSISTANT = "assistant"


# ═════════════════════════════════════════════════════════════════════════════
#  Birth Data & User
# ═════════════════════════════════════════════════════════════════════════════


class BirthData(BaseModel):
    """
    Birth information required to calculate a Tu Vi chart.

    Fields:
        name:       Full name of the person.
        gender:     Gender (Nam / Nu).
        solar_dob:  Date of birth in the Gregorian (solar) calendar.
        birth_hour: The 2-hour Chinese time period (gio sinh), 0-11 mapping
                    to Ty(0)..Hoi(11).  -1 means "unknown".
    """

    name: str = Field(..., min_length=1, max_length=200, description="Full name")
    gender: Gender
    solar_dob: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Solar date of birth, YYYY-MM-DD",
    )
    birth_hour: int = Field(
        ...,
        ge=-1,
        le=11,
        description="Chinese double-hour index (0=Ty .. 11=Hoi, -1=unknown)",
    )


class User(BaseModel):
    """
    A registered user of the chatbot.

    The ``user_id`` is the Telegram ``chat_id`` (stringified) so it
    doubles as the primary key in MongoDB.
    """

    user_id: str = Field(..., description="Unique user identifier (Telegram chat_id)")
    birth_data: BirthData | None = Field(
        default=None,
        description="Birth data collected from the user; None if not yet provided",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═════════════════════════════════════════════════════════════════════════════
#  Chat Message
# ═════════════════════════════════════════════════════════════════════════════


class ChatMessage(BaseModel):
    """
    A single message in the conversation history.

    Stored in the ``chat_histories`` MongoDB collection.
    The ``embedding`` field is populated asynchronously after the message
    is created, enabling vector search over conversation history.
    """

    user_id: str = Field(..., description="Owner of this message (Telegram chat_id)")
    role: MessageRole
    content: str = Field(..., min_length=1)
    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector embedding for long-term memory retrieval",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═════════════════════════════════════════════════════════════════════════════
#  Document Ingestion (preserved from v2)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class Chunk:
    """A text chunk from a parsed document, with associated metadata."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionResult:
    """Result of a successful document ingestion."""

    chunks_stored: int
    original_name: str
    upload_date: str


@dataclass
class AnalysisResult:
    """Result of the full OCR -> RAG -> Answer pipeline."""

    answer: str
    ocr_text: str
    chunks_used: int
