"""
main.py
══════════════════════════════════════════════════════════════════════════════
FastAPI application for the Tử Vi RAG Data Ingestion Pipeline + Telegram Bot.

Endpoints:
  POST /api/ingest
    Accepts a single file upload (field name: `document`).
    Saves the file temporarily to `uploads/`, invokes the RAG ingestion
    service, then cleans up the temp file — whether or not ingestion succeeds.

  POST /telegram/webhook
    Receives Telegram Update objects. Routes photo messages through the
    Claude Vision → RAG → Answer pipeline and replies to the user.

  GET /health
    Liveness probe.

Run the server:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Telegram webhook setup (development):
  1. Start ngrok:          ngrok http 8000
  2. Set .env:             WEBHOOK_BASE_URL=https://<ngrok-id>.ngrok.io
  3. Start the server — the webhook is registered automatically on startup.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from rag_service import ingest_document
from telegram_handler import telegram_router

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_BASE_URL: str = os.getenv("WEBHOOK_BASE_URL", "").rstrip("/")
TELEGRAM_API_BASE: str = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ── Directory Setup ────────────────────────────────────────────────────────────
UPLOADS_DIR: Path = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Supported MIME Types ───────────────────────────────────────────────────────
SUPPORTED_CONTENT_TYPES: set[str] = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/bmp",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Lifespan — register / deregister Telegram webhook automatically
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    FastAPI lifespan context manager.

    On startup:
      • If TELEGRAM_BOT_TOKEN and WEBHOOK_BASE_URL are both set, register the
        Telegram webhook so that Telegram starts POSTing updates to this server.

    On shutdown:
      • Nothing to do — Telegram will simply queue updates until the webhook
        is re-registered the next time the server starts.
    """
    # ── Register webhook on startup ───────────────────────────────────────
    if TELEGRAM_BOT_TOKEN and WEBHOOK_BASE_URL:
        webhook_url = f"{WEBHOOK_BASE_URL}/telegram/webhook"
        logger.info("Registering Telegram webhook: %s", webhook_url)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{TELEGRAM_API_BASE}/setWebhook",
                    json={
                        "url": webhook_url,
                        # Only receive message updates to reduce noise.
                        "allowed_updates": ["message", "edited_message"],
                        # Drop updates queued while the server was down.
                        "drop_pending_updates": True,
                    },
                )
                result = resp.json()
                if result.get("ok"):
                    logger.info("Telegram webhook registered successfully ✓")
                else:
                    logger.error(
                        "Failed to register Telegram webhook: %s", result.get("description")
                    )
        except Exception:
            logger.exception("Exception while registering Telegram webhook")
    else:
        missing = []
        if not TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not WEBHOOK_BASE_URL:
            missing.append("WEBHOOK_BASE_URL")
        logger.warning(
            "Telegram webhook NOT registered — missing env var(s): %s. "
            "Set them in .env and restart to enable the bot.",
            ", ".join(missing),
        )

    yield  # ── Server is running ─────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Tử Vi RAG — Data Ingestion API & Telegram Bot",
    description=(
        "Upload PDF, DOCX, or image files to be parsed, chunked, embedded "
        "with OpenAI, and stored in MongoDB Atlas Vector Search.\n\n"
        "A Telegram bot at `/telegram/webhook` accepts horoscope images, "
        "extracts text via Claude Vision, retrieves knowledge context from "
        "MongoDB, and returns a Vietnamese Tử Vi interpretation."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── Mount routers ──────────────────────────────────────────────────────────────
app.include_router(telegram_router)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — MIME type guard
# ══════════════════════════════════════════════════════════════════════════════

def _is_supported(content_type: str) -> bool:
    """Return True if the MIME type is accepted by the ingestion pipeline."""
    return (
        content_type in SUPPORTED_CONTENT_TYPES
        or content_type.startswith("image/")
    )


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/ingest
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/ingest",
    summary="Ingest a knowledge document",
    status_code=status.HTTP_200_OK,
)
async def ingest_endpoint(
    document: UploadFile = File(
        ...,
        description="PDF, DOCX, or image file to ingest into the vector store.",
    ),
) -> JSONResponse:
    """
    Ingest a knowledge document into MongoDB Atlas Vector Search.

    - Accepts **PDF**, **DOCX**, or **image** files.
    - The file is saved to `uploads/` under a unique UUID filename.
    - After ingestion (success or failure) the temp file is **always deleted**.

    Returns a JSON body:
    ```json
    {
      "status": "success",
      "message": "Document ingested successfully.",
      "data": {
        "chunks_stored": 42,
        "original_name": "tuvi_manual.pdf",
        "upload_date": "2025-04-07T10:30:00+00:00"
      }
    }
    ```
    """
    # ── 1. Early content-type validation ─────────────────────────────────────
    content_type: str = document.content_type or ""
    if not _is_supported(content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                "Accepted types: application/pdf, "
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document, image/*."
            ),
        )

    # ── 2. Build a unique temp path ──────────────────────────────────────────
    original_name: str = document.filename or "unknown_file"
    suffix: str = Path(original_name).suffix
    unique_filename: str = f"{uuid.uuid4().hex}{suffix}"
    temp_path: Path = UPLOADS_DIR / unique_filename

    # ── 3. Stream the upload to disk ──────────────────────────────────────────
    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
        logger.info(
            "Saved upload '%s' → '%s' (%d bytes)",
            original_name,
            temp_path,
            temp_path.stat().st_size,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    # ── 4. Ingest + guaranteed cleanup ───────────────────────────────────────
    try:
        result: dict = ingest_document(
            file_path=str(temp_path),
            original_name=original_name,
            content_type=content_type,
        )
        logger.info(
            "Ingestion complete for '%s': %d chunk(s) stored.",
            original_name,
            result["chunks_stored"],
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Document ingested successfully.",
                "data": result,
            },
        )

    except ValueError as exc:
        logger.warning("Validation error during ingestion: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error during ingestion of '%s'", original_name)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
                logger.info("Temp file deleted: %s", temp_path)
            except OSError as cleanup_err:
                logger.warning(
                    "Could not delete temp file '%s': %s", temp_path, cleanup_err
                )


# ══════════════════════════════════════════════════════════════════════════════
#  GET /health — lightweight liveness probe
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Health check", include_in_schema=False)
async def health_check() -> dict[str, str]:
    """Returns 200 OK when the server is running."""
    return {"status": "ok"}
