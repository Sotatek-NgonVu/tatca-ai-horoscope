"""
Unit tests for app/domain/pipeline.py — RAGPipeline

All external dependencies (LLM, embedding, vector store, Tu Vi engine)
are replaced with in-memory fakes from conftest.py.

No real database, no Anthropic API, no sentence-transformers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import LLMAuthenticationError, LLMRateLimitError, VectorStoreError
from app.domain.models import BirthData, ChatMessage, Chunk, Gender, MessageRole
from app.domain.pipeline import RAGPipeline, _PartialBirthData, _can_chi

# Import fakes from conftest (injected by pytest automatically)
from tests.conftest import (
    FakeEmbeddingService,
    FakeLLMService,
    FakeTuViEngine,
    InMemoryVectorStore,
)


# =============================================================================
#  Helpers
# =============================================================================


def make_pipeline(
    *,
    vector_store: InMemoryVectorStore | None = None,
    llm: FakeLLMService | None = None,
    embedding: FakeEmbeddingService | None = None,
    tuvi_engine: FakeTuViEngine | None = None,
    loader_registry: Any | None = None,
) -> RAGPipeline:
    return RAGPipeline(
        vector_store=vector_store or InMemoryVectorStore(),
        llm=llm or FakeLLMService(),
        embedding=embedding or FakeEmbeddingService(),
        tuvi_engine=tuvi_engine or FakeTuViEngine(),
        loader_registry=loader_registry,
    )


def make_birth_data(
    name: str = "Test User",
    gender: Gender = Gender.MALE,
    solar_dob: str = "1990-05-15",
    birth_hour: int = 0,
) -> BirthData:
    return BirthData(name=name, gender=gender, solar_dob=solar_dob, birth_hour=birth_hour)


# =============================================================================
#  _can_chi() (module-level helper)
# =============================================================================


class TestCanChiHelper:
    def test_1990_returns_canh_ngo(self) -> None:
        assert _can_chi(1990) == "Canh Ngo"

    def test_2000_returns_canh_thin(self) -> None:
        assert _can_chi(2000) == "Canh Thin"

    def test_cycle_repeats_every_60(self) -> None:
        assert _can_chi(1960) == _can_chi(2020)


# =============================================================================
#  _PartialBirthData
# =============================================================================


class TestPartialBirthData:
    def test_all_missing_when_empty(self) -> None:
        partial = _PartialBirthData()
        assert len(partial.missing_fields) == 4

    def test_is_complete_false_when_empty(self) -> None:
        assert _PartialBirthData().is_complete is False

    def test_is_complete_true_when_all_fields_set(self) -> None:
        partial = _PartialBirthData(
            name="Alice",
            gender=Gender.FEMALE,
            solar_dob="1995-01-01",
            birth_hour=3,
        )
        assert partial.is_complete is True
        assert partial.missing_fields == []

    def test_birth_hour_minus_one_counts_as_complete(self) -> None:
        """birth_hour=-1 means unknown, but it IS provided."""
        partial = _PartialBirthData(
            name="Bob",
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=-1,
        )
        assert partial.is_complete is True

    def test_missing_name_reported(self) -> None:
        partial = _PartialBirthData(
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=0,
        )
        labels = partial.missing_fields
        assert any("ten" in f.lower() or "Ho ten" in f for f in labels)

    def test_missing_gender_reported(self) -> None:
        partial = _PartialBirthData(
            name="Alice",
            solar_dob="1990-01-01",
            birth_hour=0,
        )
        labels = partial.missing_fields
        assert any("tinh" in f.lower() or "Gioi" in f for f in labels)

    def test_to_birth_data_succeeds_when_complete(self) -> None:
        partial = _PartialBirthData(
            name="Nguyen Van A",
            gender=Gender.MALE,
            solar_dob="1990-05-15",
            birth_hour=0,
        )
        bd = partial.to_birth_data()
        assert isinstance(bd, BirthData)
        assert bd.name == "Nguyen Van A"

    def test_to_birth_data_raises_when_incomplete(self) -> None:
        partial = _PartialBirthData(name="Incomplete")
        with pytest.raises(AssertionError):
            partial.to_birth_data()

    def test_merge_fills_missing_fields(self) -> None:
        base = _PartialBirthData(name="Alice")
        other = _PartialBirthData(
            gender=Gender.FEMALE,
            solar_dob="1995-01-01",
            birth_hour=5,
        )
        base.merge(other)
        assert base.name == "Alice"        # kept
        assert base.gender == Gender.FEMALE
        assert base.solar_dob == "1995-01-01"
        assert base.birth_hour == 5

    def test_merge_does_not_overwrite_existing_values(self) -> None:
        base = _PartialBirthData(name="Alice", gender=Gender.FEMALE)
        other = _PartialBirthData(name="Bob", gender=Gender.MALE)
        base.merge(other)
        # Existing values kept
        assert base.name == "Alice"
        assert base.gender == Gender.FEMALE


# =============================================================================
#  RAGPipeline.ingest()
# =============================================================================


class TestRAGPipelineIngest:
    def test_ingest_raises_without_loader_registry(self) -> None:
        pipeline = make_pipeline(loader_registry=None)
        with pytest.raises(RuntimeError, match="loader_registry"):
            pipeline.ingest("/path/to/file.pdf", "tuvi.pdf", "application/pdf")

    def test_ingest_stores_chunks_and_returns_result(self) -> None:
        store = InMemoryVectorStore()

        class FakeRegistry:
            def load(self, file_path: str, content_type: str) -> list[Chunk]:
                return [
                    Chunk(content="Chunk 1 text"),
                    Chunk(content="Chunk 2 text"),
                ]

        pipeline = make_pipeline(
            vector_store=store,
            loader_registry=FakeRegistry(),
        )
        result = pipeline.ingest("/fake/path.pdf", "tuvi_guide.pdf", "application/pdf")

        assert result.original_name == "tuvi_guide.pdf"
        assert result.chunks_stored > 0

    def test_ingest_enriches_chunk_metadata(self) -> None:
        store = InMemoryVectorStore()

        class FakeRegistry:
            def load(self, file_path: str, content_type: str) -> list[Chunk]:
                return [Chunk(content="Text content")]

        pipeline = make_pipeline(
            vector_store=store,
            loader_registry=FakeRegistry(),
        )
        pipeline.ingest("/fake/path.pdf", "my_doc.pdf", "application/pdf")

        assert len(store._documents) > 0
        # Every stored chunk should have the source enriched
        for chunk in store._documents:
            assert chunk.metadata.get("source") == "my_doc.pdf"
            assert "upload_date" in chunk.metadata


# =============================================================================
#  RAGPipeline.chat() — happy path (user has birth data)
# =============================================================================


class TestRAGPipelineChatHappyPath:
    def test_chat_returns_llm_answer(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService(response="Day la cau tra loi cua Tu Vi AI")

        # Pre-seed the user with birth data
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        pipeline = make_pipeline(vector_store=store, llm=llm)
        answer = pipeline.chat("user_1", "Menh toi la gi?")

        assert answer == "Day la cau tra loi cua Tu Vi AI"

    def test_chat_calls_llm_generate(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        pipeline = make_pipeline(vector_store=store, llm=llm)
        pipeline.chat("user_1", "Boi kien thuc Tu Vi")

        assert len(llm.generate_calls) == 1

    def test_chat_persists_user_and_assistant_messages(self) -> None:
        store = InMemoryVectorStore()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        pipeline = make_pipeline(vector_store=store)
        pipeline.chat("user_1", "My query")

        user_msgs = [m for m in store._messages if m.role == MessageRole.USER]
        asst_msgs = [m for m in store._messages if m.role == MessageRole.ASSISTANT]
        assert len(user_msgs) == 1
        assert len(asst_msgs) == 1

    def test_chat_generates_tuvi_chart(self) -> None:
        store = InMemoryVectorStore()
        engine = FakeTuViEngine()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        pipeline = make_pipeline(vector_store=store, tuvi_engine=engine)
        pipeline.chat("user_1", "Some question")

        assert engine.call_count == 1
        assert engine.last_birth_data is not None

    def test_chat_returns_fallback_when_llm_returns_empty(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService(response="")
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        pipeline = make_pipeline(vector_store=store, llm=llm)
        answer = pipeline.chat("user_1", "My query")

        assert len(answer) > 0
        assert "loi" in answer.lower() or "thu lai" in answer.lower()


# =============================================================================
#  RAGPipeline.chat() — error handling
# =============================================================================


class TestRAGPipelineChatErrorHandling:
    def test_llm_auth_error_returns_friendly_message(self) -> None:
        store = InMemoryVectorStore()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        class AuthFailLLM(FakeLLMService):
            def generate(self, *, system_prompt, user_message, max_tokens=4096):
                raise LLMAuthenticationError("bad key")

        pipeline = make_pipeline(vector_store=store, llm=AuthFailLLM())
        answer = pipeline.chat("user_1", "Hello")

        assert "API" in answer or "xac thuc" in answer.lower()

    def test_llm_rate_limit_error_returns_friendly_message(self) -> None:
        store = InMemoryVectorStore()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        class RateLimitLLM(FakeLLMService):
            def generate(self, *, system_prompt, user_message, max_tokens=4096):
                raise LLMRateLimitError("too many requests")

        pipeline = make_pipeline(vector_store=store, llm=RateLimitLLM())
        answer = pipeline.chat("user_1", "Hello")

        assert "ban" in answer.lower() or "thu lai" in answer.lower()

    def test_unexpected_error_returns_friendly_message(self) -> None:
        store = InMemoryVectorStore()
        bd = make_birth_data()
        store.upsert_user("user_1", {"birth_data": bd.model_dump()})

        class BrokenLLM(FakeLLMService):
            def generate(self, *, system_prompt, user_message, max_tokens=4096):
                raise RuntimeError("totally unexpected")

        pipeline = make_pipeline(vector_store=store, llm=BrokenLLM())
        answer = pipeline.chat("user_1", "Hello")

        assert len(answer) > 0


# =============================================================================
#  RAGPipeline.chat() — birth data collection flow
# =============================================================================


class TestRAGPipelineBirthDataCollection:
    def test_returns_birth_data_prompt_on_first_interaction_with_no_data(self) -> None:
        llm = FakeLLMService()
        llm.set_extract_response(None)  # LLM finds nothing

        pipeline = make_pipeline(llm=llm)
        answer = pipeline.chat("new_user", "Xin chao")

        # Should ask for birth data
        assert len(answer) > 0
        # The pipeline prompts for birth data
        assert "sinh" in answer.lower() or "thong tin" in answer.lower() or "chao" in answer.lower()

    def test_complete_birth_data_in_first_message_saves_and_responds(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService(response="Xin chao! Day la tong quan la so cua ban.")
        llm.set_extract_response({
            "name": "Nguyen Van A",
            "gender": "Nam",
            "solar_dob": "1990-05-15",
            "birth_hour": 0,
        })

        pipeline = make_pipeline(vector_store=store, llm=llm)
        answer = pipeline.chat("new_user", "Nguyen Van A, Nam, 15/05/1990, gio Ty")

        # Birth data should be saved
        user_doc = store.get_user("new_user")
        assert user_doc is not None
        assert "birth_data" in user_doc
        # Response should not be the raw birth data prompt
        assert len(answer) > 0

    def test_partial_birth_data_accumulates_across_turns(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService()

        # First turn: provide name only
        llm.set_extract_response({
            "name": "Alice",
            "gender": None,
            "solar_dob": None,
            "birth_hour": None,
        })
        pipeline = make_pipeline(vector_store=store, llm=llm)
        answer_1 = pipeline.chat("user_partial", "Ten toi la Alice")

        # Should still be collecting
        assert "Alice" in answer_1 or "ten" in answer_1.lower() or "thong tin" in answer_1.lower()

        # Second turn: provide remaining fields
        llm.set_extract_response({
            "name": None,
            "gender": "Nu",
            "solar_dob": "1995-08-20",
            "birth_hour": 3,
        })
        llm.response = "Chao Alice! Day la la so Tu Vi cua ban."
        answer_2 = pipeline.chat("user_partial", "Nu, 20/08/1995, gio Mao")

        # Birth data should be complete and saved
        user_doc = store.get_user("user_partial")
        assert user_doc is not None
        bd = user_doc.get("birth_data", {})
        assert bd.get("name") == "Alice"

    def test_pending_cleared_after_complete_birth_data(self) -> None:
        store = InMemoryVectorStore()
        llm = FakeLLMService()

        # Seed partial state
        llm.set_extract_response({
            "name": "Bob",
            "gender": None,
            "solar_dob": None,
            "birth_hour": None,
        })
        pipeline = make_pipeline(vector_store=store, llm=llm)
        pipeline.chat("user_bob", "Ten toi la Bob")

        # Complete it
        llm.set_extract_response({
            "name": None,
            "gender": "Nam",
            "solar_dob": "1988-03-12",
            "birth_hour": 7,
        })
        llm.response = "Chao Bob!"
        pipeline.chat("user_bob", "Nam, 12/03/1988, gio Mui")

        # The _pending dict should no longer have this user
        assert "user_bob" not in pipeline._pending


# =============================================================================
#  RAGPipeline._reprompt_missing()
# =============================================================================


class TestRepromptMissing:
    def test_reprompt_lists_all_missing_when_empty(self) -> None:
        pipeline = make_pipeline()
        partial = _PartialBirthData()
        response = pipeline._reprompt_missing(partial)

        assert "Ho ten" in response
        assert "Gioi tinh" in response
        assert "Ngay sinh" in response
        assert "Gio sinh" in response

    def test_reprompt_acknowledges_collected_name(self) -> None:
        pipeline = make_pipeline()
        partial = _PartialBirthData(name="Tran Thi C")
        response = pipeline._reprompt_missing(partial)

        assert "Tran Thi C" in response

    def test_reprompt_shows_gender_if_collected(self) -> None:
        pipeline = make_pipeline()
        partial = _PartialBirthData(
            name="Alice",
            gender=Gender.FEMALE,
        )
        response = pipeline._reprompt_missing(partial)
        assert "Nu" in response

    def test_reprompt_shows_unknown_birth_hour(self) -> None:
        pipeline = make_pipeline()
        partial = _PartialBirthData(
            name="Bob",
            gender=Gender.MALE,
            solar_dob="1990-01-01",
            birth_hour=-1,
        )
        response = pipeline._reprompt_missing(partial)
        assert "khong ro" in response.lower()

    def test_reprompt_shows_hour_name_for_known_hour(self) -> None:
        pipeline = make_pipeline()
        partial = _PartialBirthData(
            name="Carol",
            gender=Gender.FEMALE,
            solar_dob="1990-01-01",
            birth_hour=0,  # Ty
        )
        response = pipeline._reprompt_missing(partial)
        assert "Ty" in response


# =============================================================================
#  RAGPipeline._generate_answer() context assembly
# =============================================================================


class TestGenerateAnswer:
    def test_assembles_prompt_with_chart(self) -> None:
        llm = FakeLLMService()
        pipeline = make_pipeline(llm=llm)
        pipeline._generate_answer(
            query="My question",
            chart_json={"menh_cung": "Thien Di"},
            short_term=[],
            long_term=[],
        )
        assert len(llm.generate_calls) == 1
        user_message = llm.generate_calls[0]["user_message"]
        assert "La so Tu Vi" in user_message or "Tu Vi" in user_message

    def test_assembles_prompt_with_short_term_history(self) -> None:
        llm = FakeLLMService()
        pipeline = make_pipeline(llm=llm)
        history = [
            ChatMessage(user_id="u", role=MessageRole.USER, content="Previous question"),
            ChatMessage(user_id="u", role=MessageRole.ASSISTANT, content="Previous answer"),
        ]
        pipeline._generate_answer(
            query="New question",
            chart_json={},
            short_term=history,
            long_term=[],
        )
        user_message = llm.generate_calls[0]["user_message"]
        assert "Previous question" in user_message
        assert "Previous answer" in user_message

    def test_assembles_prompt_with_long_term_memory(self) -> None:
        llm = FakeLLMService()
        pipeline = make_pipeline(llm=llm)
        lt = [
            ChatMessage(user_id="u", role=MessageRole.USER, content="Long ago memory"),
        ]
        pipeline._generate_answer(
            query="Current question",
            chart_json={},
            short_term=[],
            long_term=lt,
        )
        user_message = llm.generate_calls[0]["user_message"]
        assert "Long ago memory" in user_message

    def test_query_always_appears_in_prompt(self) -> None:
        llm = FakeLLMService()
        pipeline = make_pipeline(llm=llm)
        pipeline._generate_answer(
            query="What is menh cung?",
            chart_json={},
            short_term=[],
            long_term=[],
        )
        user_message = llm.generate_calls[0]["user_message"]
        assert "What is menh cung?" in user_message

    def test_system_prompt_contains_current_year(self) -> None:
        from datetime import datetime, timezone
        llm = FakeLLMService()
        pipeline = make_pipeline(llm=llm)
        pipeline._generate_answer(
            query="question",
            chart_json={},
            short_term=[],
            long_term=[],
        )
        system_prompt = llm.generate_calls[0]["system_prompt"]
        current_year = str(datetime.now(tz=timezone.utc).year)
        assert current_year in system_prompt


# =============================================================================
#  RAGPipeline._persist_messages()
# =============================================================================


class TestPersistMessages:
    def test_persist_saves_two_messages(self) -> None:
        store = InMemoryVectorStore()
        pipeline = make_pipeline(vector_store=store)
        pipeline._persist_messages("user_1", "query text", "answer text")

        assert len(store._messages) == 2
        roles = {m.role for m in store._messages}
        assert MessageRole.USER in roles
        assert MessageRole.ASSISTANT in roles

    def test_persist_stores_correct_content(self) -> None:
        store = InMemoryVectorStore()
        pipeline = make_pipeline(vector_store=store)
        pipeline._persist_messages("user_1", "my query", "my answer")

        contents = {m.content for m in store._messages}
        assert "my query" in contents
        assert "my answer" in contents

    def test_persist_failure_does_not_raise(self) -> None:
        """_persist_messages is best-effort; errors must not propagate."""

        class FailingStore(InMemoryVectorStore):
            def save_chat_message(self, message: ChatMessage) -> None:
                raise RuntimeError("disk full")

        pipeline = make_pipeline(vector_store=FailingStore())
        # Must not raise
        pipeline._persist_messages("user_1", "query", "answer")


# =============================================================================
#  RAGPipeline.save_birth_data() / _get_birth_data()
# =============================================================================


class TestBirthDataPersistence:
    def test_save_and_retrieve_birth_data(self) -> None:
        store = InMemoryVectorStore()
        pipeline = make_pipeline(vector_store=store)
        bd = make_birth_data(name="Stored User")

        pipeline.save_birth_data("user_x", bd)
        retrieved = pipeline._get_birth_data("user_x")

        assert retrieved is not None
        assert retrieved.name == "Stored User"

    def test_get_birth_data_returns_none_for_unknown_user(self) -> None:
        store = InMemoryVectorStore()
        pipeline = make_pipeline(vector_store=store)
        result = pipeline._get_birth_data("nonexistent_user")
        assert result is None

    def test_get_birth_data_returns_none_when_user_has_no_birth_data(self) -> None:
        store = InMemoryVectorStore()
        store.upsert_user("user_no_bd", {"some_other_field": "value"})
        pipeline = make_pipeline(vector_store=store)
        result = pipeline._get_birth_data("user_no_bd")
        assert result is None


# =============================================================================
#  RAGPipeline._fetch_long_term_memory()
# =============================================================================


class TestFetchLongTermMemory:
    def test_empty_query_returns_empty_list(self) -> None:
        pipeline = make_pipeline()
        result = pipeline._fetch_long_term_memory("user_1", "   ")
        assert result == []

    def test_returns_messages_from_store(self) -> None:
        store = InMemoryVectorStore()
        msg = ChatMessage(
            user_id="user_1", role=MessageRole.USER, content="Old message"
        )
        store.save_chat_message(msg)
        pipeline = make_pipeline(vector_store=store)

        result = pipeline._fetch_long_term_memory("user_1", "some query")
        assert len(result) >= 1


# =============================================================================
#  RAGPipeline._fetch_short_term_memory()
# =============================================================================


class TestFetchShortTermMemory:
    def test_returns_empty_when_no_messages(self) -> None:
        pipeline = make_pipeline()
        result = pipeline._fetch_short_term_memory("user_1")
        assert result == []

    def test_returns_recent_messages(self) -> None:
        store = InMemoryVectorStore()
        for i in range(3):
            store.save_chat_message(
                ChatMessage(user_id="u1", role=MessageRole.USER, content=f"msg {i}")
            )
        pipeline = make_pipeline(vector_store=store)
        result = pipeline._fetch_short_term_memory("u1")
        assert len(result) == 3

    def test_vector_store_error_returns_empty_list(self) -> None:
        class FailingStore(InMemoryVectorStore):
            def get_recent_messages(self, user_id: str, *, limit: int = 5):
                raise VectorStoreError("db down")

        pipeline = make_pipeline(vector_store=FailingStore())
        result = pipeline._fetch_short_term_memory("user_1")
        assert result == []
