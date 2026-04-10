"""
FastAPI application factory.

Creates and configures the FastAPI app with:
  - Centralized configuration via Pydantic Settings
  - Lifespan management (DB connection, Telegram webhook)
  - Router mounting (API routes + Telegram bot)
  - Exception handlers for domain errors

Run:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config.settings import get_settings
from app.core.exceptions import (
    AppError,
    BirthDataMissingError,
    DocumentLoadError,
    UnsupportedDocumentTypeError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s --- %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Lifespan --- startup / shutdown hooks
# ══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan.

    On startup:
      1. Validate configuration
      2. Connect to MongoDB
      3. Ensure required collections exist
      4. Register Telegram webhook (if configured)

    On shutdown:
      1. Close MongoDB connection
    """
    from app.bot.telegram.client import TelegramClient
    from app.bot.telegram.router import set_telegram_client
    from app.infrastructure.dependencies import (
        init_database_manager,
        shutdown_database_manager,
    )
    from app.infrastructure.setup import ensure_collections_and_indexes

    settings = get_settings()

    # ── 1. Connect to MongoDB ────────────────────────────────────────────
    logger.info("Initializing database connection...")
    db_manager = init_database_manager(settings)

    # ── 2. Ensure required collections and indexes exist ─────────────────
    ensure_collections_and_indexes(db_manager.get_database(), settings)

    # Keep a reference so we can close the HTTP client on shutdown.
    _telegram_client_ref: TelegramClient | None = None

    # ── 3. Register Telegram webhook ─────────────────────────────────────
    if settings.TELEGRAM_BOT_TOKEN and settings.WEBHOOK_BASE_URL:
        telegram_client = TelegramClient(settings.TELEGRAM_BOT_TOKEN)
        _telegram_client_ref = telegram_client
        set_telegram_client(telegram_client)

        webhook_url = f"{settings.WEBHOOK_BASE_URL.rstrip('/')}/telegram/webhook"
        await telegram_client.register_webhook(webhook_url)
    else:
        missing = []
        if not settings.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not settings.WEBHOOK_BASE_URL:
            missing.append("WEBHOOK_BASE_URL")
        logger.warning(
            "Telegram webhook NOT registered --- missing: %s",
            ", ".join(missing),
        )

    yield  # ── Server is running ─────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    if _telegram_client_ref is not None:
        await _telegram_client_ref.aclose()
    shutdown_database_manager()


# ══════════════════════════════════════════════════════════════════════════════
#  Application Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="Tu Vi RAG Chatbot v3.0",
        description=(
            "Vietnamese astrology (Tu Vi) AI chatbot powered by Claude, "
            "with hybrid memory (short-term + long-term vector search), "
            "backed by MongoDB Atlas.\n\n"
            "Telegram bot at `/telegram/webhook` accepts text messages "
            "and horoscope images."
        ),
        version="3.0.0",
        lifespan=lifespan,
    )

    # ── Mount routers ────────────────────────────────────────────────────
    from app.api.routes.health import health_router
    from app.api.routes.ingest import ingest_router
    from app.bot.telegram.router import telegram_router

    application.include_router(health_router)
    application.include_router(ingest_router)
    application.include_router(telegram_router)

    # ── Exception handlers ───────────────────────────────────────────────

    @application.exception_handler(BirthDataMissingError)
    async def birth_data_missing_handler(
        request: Request, exc: BirthDataMissingError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.message},
        )

    @application.exception_handler(UnsupportedDocumentTypeError)
    async def unsupported_doc_handler(
        request: Request, exc: UnsupportedDocumentTypeError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=415,
            content={"detail": exc.message},
        )

    @application.exception_handler(DocumentLoadError)
    async def doc_load_handler(
        request: Request, exc: DocumentLoadError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.message},
        )

    @application.exception_handler(AppError)
    async def app_error_handler(
        request: Request, exc: AppError
    ) -> JSONResponse:
        logger.error("Unhandled AppError: %s", exc.message)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return application


# Module-level app instance for uvicorn
app = create_app()
