"""
Unit tests for app/domain/models.py

Tests all domain model validation, field constraints, and enum behaviour.
No database, no API calls — pure Python logic only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models import (
    AnalysisResult,
    BirthData,
    ChatMessage,
    Chunk,
    Gender,
    IngestionResult,
    MessageRole,
    User,
)


# =============================================================================
#  Gender
# =============================================================================


class TestGender:
    def test_male_value(self) -> None:
        assert Gender.MALE == "Nam"

    def test_female_value(self) -> None:
        assert Gender.FEMALE == "Nu"

    def test_is_str_enum(self) -> None:
        assert isinstance(Gender.MALE, str)


# =============================================================================
#  MessageRole
# =============================================================================


class TestMessageRole:
    def test_user_value(self) -> None:
        assert MessageRole.USER == "user"

    def test_assistant_value(self) -> None:
        assert MessageRole.ASSISTANT == "assistant"


# =============================================================================
#  BirthData
# =============================================================================


class TestBirthData:
    def test_valid_birth_data(self) -> None:
        bd = BirthData(
            name="Tran Thi B",
            gender=Gender.FEMALE,
            solar_dob="1995-08-20",
            birth_hour=3,
        )
        assert bd.name == "Tran Thi B"
        assert bd.gender == Gender.FEMALE
        assert bd.solar_dob == "1995-08-20"
        assert bd.birth_hour == 3

    def test_name_min_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BirthData(
                name="",
                gender=Gender.MALE,
                solar_dob="1990-01-01",
                birth_hour=0,
            )

    def test_name_max_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BirthData(
                name="A" * 201,
                gender=Gender.MALE,
                solar_dob="1990-01-01",
                birth_hour=0,
            )

    def test_solar_dob_pattern_enforced(self) -> None:
        """Dates outside YYYY-MM-DD format are rejected."""
        with pytest.raises(ValidationError):
            BirthData(
                name="Valid Name",
                gender=Gender.MALE,
                solar_dob="15/05/1990",  # wrong format
                birth_hour=0,
            )

    def test_birth_hour_minus_one_allowed(self) -> None:
        """birth_hour=-1 means unknown."""
        bd = BirthData(
            name="Valid Name",
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=-1,
        )
        assert bd.birth_hour == -1

    def test_birth_hour_zero_allowed(self) -> None:
        bd = BirthData(
            name="Valid Name",
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=0,
        )
        assert bd.birth_hour == 0

    def test_birth_hour_eleven_allowed(self) -> None:
        bd = BirthData(
            name="Valid Name",
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=11,
        )
        assert bd.birth_hour == 11

    def test_birth_hour_below_minus_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BirthData(
                name="Valid Name",
                gender=Gender.MALE,
                solar_dob="1990-01-01",
                birth_hour=-2,
            )

    def test_birth_hour_above_eleven_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BirthData(
                name="Valid Name",
                gender=Gender.MALE,
                solar_dob="1990-01-01",
                birth_hour=12,
            )

    def test_invalid_gender_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BirthData(
                name="Valid Name",
                gender="Unknown",  # type: ignore[arg-type]
                solar_dob="1990-01-01",
                birth_hour=0,
            )


# =============================================================================
#  User
# =============================================================================


class TestUser:
    def test_user_without_birth_data(self) -> None:
        user = User(user_id="telegram_123")
        assert user.user_id == "telegram_123"
        assert user.birth_data is None

    def test_user_with_birth_data(self) -> None:
        bd = BirthData(
            name="Le Van C",
            gender=Gender.MALE,
            solar_dob="1985-03-10",
            birth_hour=5,
        )
        user = User(user_id="telegram_456", birth_data=bd)
        assert user.birth_data == bd

    def test_created_at_set_automatically(self) -> None:
        user = User(user_id="telegram_789")
        assert user.created_at is not None

    def test_updated_at_set_automatically(self) -> None:
        user = User(user_id="telegram_789")
        assert user.updated_at is not None


# =============================================================================
#  ChatMessage
# =============================================================================


class TestChatMessage:
    def test_valid_chat_message(self) -> None:
        msg = ChatMessage(
            user_id="user_1",
            role=MessageRole.USER,
            content="Hello, Tu Vi AI!",
        )
        assert msg.user_id == "user_1"
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, Tu Vi AI!"
        assert msg.embedding is None

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(
                user_id="user_1",
                role=MessageRole.USER,
                content="",
            )

    def test_assistant_role(self) -> None:
        msg = ChatMessage(
            user_id="user_1",
            role=MessageRole.ASSISTANT,
            content="Xin chao!",
        )
        assert msg.role == MessageRole.ASSISTANT

    def test_embedding_optional(self) -> None:
        msg = ChatMessage(
            user_id="user_1",
            role=MessageRole.USER,
            content="Test message",
            embedding=[0.1, 0.2, 0.3],
        )
        assert msg.embedding == [0.1, 0.2, 0.3]

    def test_metadata_defaults_to_empty_dict(self) -> None:
        msg = ChatMessage(
            user_id="user_1",
            role=MessageRole.USER,
            content="Test",
        )
        assert msg.metadata == {}

    def test_created_at_auto_populated(self) -> None:
        msg = ChatMessage(
            user_id="user_1",
            role=MessageRole.USER,
            content="Test",
        )
        assert msg.created_at is not None


# =============================================================================
#  Chunk (dataclass)
# =============================================================================


class TestChunk:
    def test_chunk_with_content_only(self) -> None:
        chunk = Chunk(content="Some text")
        assert chunk.content == "Some text"
        assert chunk.metadata == {}

    def test_chunk_with_metadata(self) -> None:
        chunk = Chunk(content="Text", metadata={"source": "test.pdf", "page": 1})
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1


# =============================================================================
#  IngestionResult (dataclass)
# =============================================================================


class TestIngestionResult:
    def test_ingestion_result_fields(self) -> None:
        result = IngestionResult(
            chunks_stored=42,
            original_name="tuvi_guide.pdf",
            upload_date="2025-01-01T00:00:00+00:00",
        )
        assert result.chunks_stored == 42
        assert result.original_name == "tuvi_guide.pdf"
        assert result.upload_date == "2025-01-01T00:00:00+00:00"


# =============================================================================
#  AnalysisResult (dataclass)
# =============================================================================


class TestAnalysisResult:
    def test_analysis_result_fields(self) -> None:
        result = AnalysisResult(
            answer="This is the answer",
            ocr_text="Extracted OCR text",
            chunks_used=3,
        )
        assert result.answer == "This is the answer"
        assert result.ocr_text == "Extracted OCR text"
        assert result.chunks_used == 3
