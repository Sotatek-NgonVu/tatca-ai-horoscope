"""
Integration tests for FastAPI endpoints.

Uses FastAPI TestClient with mocked services (no real DB or API calls).
We build a minimal test app that uses the same routers but skips the
production lifespan (which requires real env vars / DB connection).
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.health import health_router
from app.api.routes.ingest import ingest_router
from app.bot.telegram.router import set_telegram_client, telegram_router
from app.domain.pipeline import RAGPipeline
from app.infrastructure.dependencies import get_rag_pipeline
from app.services.document_loader import DocumentLoaderRegistry
from tests.conftest import (
    MockDocumentLoader,
    MockLLMService,
    MockOCRService,
    MockVectorStore,
)


@pytest.fixture
def mock_pipeline() -> RAGPipeline:
    """Create a pipeline with all mocked services."""
    registry = DocumentLoaderRegistry()
    registry.register(MockDocumentLoader())

    return RAGPipeline(
        vector_store=MockVectorStore(),
        llm=MockLLMService(),
        ocr=MockOCRService(),
        loader_registry=registry,
    )


@pytest.fixture
def test_app(mock_pipeline: RAGPipeline) -> FastAPI:
    """
    Create a lightweight FastAPI app for testing.

    Uses the real routers but no lifespan — avoids needing env vars or DB.
    """
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(ingest_router)
    app.include_router(telegram_router)

    # Override the pipeline dependency
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

    # Set up a mock Telegram client so the webhook handler doesn't crash
    mock_tg_client = MagicMock()
    mock_tg_client.send_message = AsyncMock()
    mock_tg_client.download_file = AsyncMock(return_value=b"fake-image")
    set_telegram_client(mock_tg_client)

    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """Create a test client."""
    with TestClient(test_app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestIngestEndpoint:
    """Tests for POST /api/ingest."""

    def test_ingest_rejects_unsupported_type(self, client: TestClient) -> None:
        """Uploading an unsupported file type should return 415."""
        file_data = io.BytesIO(b"not a real file")
        response = client.post(
            "/api/ingest",
            files={"document": ("test.xyz", file_data, "application/octet-stream")},
        )
        assert response.status_code == 415

    def test_ingest_accepts_pdf(self, client: TestClient) -> None:
        """Uploading a PDF should be accepted (mock loader handles it)."""
        file_data = io.BytesIO(b"%PDF-1.4 fake pdf content")
        response = client.post(
            "/api/ingest",
            files={"document": ("test.pdf", file_data, "application/pdf")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["data"]["original_name"] == "test.pdf"

    def test_ingest_accepts_image(self, client: TestClient) -> None:
        """Uploading an image should be accepted."""
        file_data = io.BytesIO(b"\x89PNG fake image content")
        response = client.post(
            "/api/ingest",
            files={"document": ("chart.png", file_data, "image/png")},
        )
        assert response.status_code == 200


class TestTelegramWebhook:
    """Tests for POST /telegram/webhook."""

    def test_empty_update_returns_ok(self, client: TestClient) -> None:
        """An update with no message should return 200."""
        response = client.post("/telegram/webhook", json={"update_id": 1})
        assert response.status_code == 200
        assert response.json() == {"ok": True}
