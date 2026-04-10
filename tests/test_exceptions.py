"""
Unit tests for app/core/exceptions.py

Tests the custom exception hierarchy: instantiation, message,
cause chaining, and inheritance relationships.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    AppError,
    BirthDataMissingError,
    DocumentLoadError,
    EmbeddingError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    OCRError,
    TuViEngineError,
    UnsupportedDocumentTypeError,
    VectorStoreError,
)


# =============================================================================
#  AppError (base)
# =============================================================================


class TestAppError:
    def test_message_stored(self) -> None:
        err = AppError("Something went wrong")
        assert err.message == "Something went wrong"

    def test_cause_defaults_to_none(self) -> None:
        err = AppError("msg")
        assert err.cause is None

    def test_cause_can_be_set(self) -> None:
        root = ValueError("root cause")
        err = AppError("wrapper", cause=root)
        assert err.cause is root

    def test_str_representation(self) -> None:
        err = AppError("test message")
        assert str(err) == "test message"

    def test_is_exception(self) -> None:
        err = AppError("msg")
        assert isinstance(err, Exception)


# =============================================================================
#  DocumentLoadError
# =============================================================================


class TestDocumentLoadError:
    def test_inherits_app_error(self) -> None:
        err = DocumentLoadError("load failed")
        assert isinstance(err, AppError)

    def test_message(self) -> None:
        err = DocumentLoadError("Cannot read file")
        assert err.message == "Cannot read file"


# =============================================================================
#  UnsupportedDocumentTypeError
# =============================================================================


class TestUnsupportedDocumentTypeError:
    def test_inherits_document_load_error(self) -> None:
        err = UnsupportedDocumentTypeError("bad type")
        assert isinstance(err, DocumentLoadError)
        assert isinstance(err, AppError)

    def test_message(self) -> None:
        err = UnsupportedDocumentTypeError("application/zip not supported")
        assert "application/zip" in err.message


# =============================================================================
#  EmbeddingError
# =============================================================================


class TestEmbeddingError:
    def test_inherits_app_error(self) -> None:
        err = EmbeddingError("embed failed")
        assert isinstance(err, AppError)

    def test_cause_chain(self) -> None:
        root = RuntimeError("model load failed")
        err = EmbeddingError("Cannot embed", cause=root)
        assert err.cause is root


# =============================================================================
#  VectorStoreError
# =============================================================================


class TestVectorStoreError:
    def test_inherits_app_error(self) -> None:
        err = VectorStoreError("store failed")
        assert isinstance(err, AppError)


# =============================================================================
#  LLMError hierarchy
# =============================================================================


class TestLLMErrorHierarchy:
    def test_llm_error_inherits_app_error(self) -> None:
        err = LLMError("llm failed")
        assert isinstance(err, AppError)

    def test_llm_auth_error_inherits_llm_error(self) -> None:
        err = LLMAuthenticationError("auth failed")
        assert isinstance(err, LLMError)
        assert isinstance(err, AppError)

    def test_llm_rate_limit_error_inherits_llm_error(self) -> None:
        err = LLMRateLimitError("rate limit")
        assert isinstance(err, LLMError)
        assert isinstance(err, AppError)

    def test_auth_error_message(self) -> None:
        err = LLMAuthenticationError("Invalid API key")
        assert "Invalid API key" in err.message

    def test_rate_limit_error_message(self) -> None:
        err = LLMRateLimitError("Too many requests")
        assert "Too many requests" in err.message


# =============================================================================
#  OCRError
# =============================================================================


class TestOCRError:
    def test_inherits_app_error(self) -> None:
        err = OCRError("ocr failed")
        assert isinstance(err, AppError)


# =============================================================================
#  TuViEngineError
# =============================================================================


class TestTuViEngineError:
    def test_inherits_app_error(self) -> None:
        err = TuViEngineError("chart failed")
        assert isinstance(err, AppError)

    def test_can_raise_and_catch_as_app_error(self) -> None:
        with pytest.raises(AppError):
            raise TuViEngineError("Test error")


# =============================================================================
#  BirthDataMissingError
# =============================================================================


class TestBirthDataMissingError:
    def test_inherits_app_error(self) -> None:
        err = BirthDataMissingError("missing birth data")
        assert isinstance(err, AppError)

    def test_can_raise_and_catch(self) -> None:
        with pytest.raises(BirthDataMissingError):
            raise BirthDataMissingError("User has not provided birth data")
