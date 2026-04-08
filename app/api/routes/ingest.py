"""
Document ingestion endpoint.

POST /api/ingest — accepts a file upload, ingests it into the vector store.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.domain.pipeline import RAGPipeline
from app.infrastructure.dependencies import get_rag_pipeline

logger = logging.getLogger(__name__)

ingest_router = APIRouter(prefix="/api", tags=["Ingestion"])

# ── Upload directory ─────────────────────────────────────────────────────────
UPLOADS_DIR: Path = Path(__file__).resolve().parents[3] / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Supported MIME types ─────────────────────────────────────────────────────
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


def _is_supported(content_type: str) -> bool:
    """Return True if the MIME type is accepted by the ingestion pipeline."""
    return content_type in SUPPORTED_CONTENT_TYPES or content_type.startswith("image/")


@ingest_router.post(
    "/ingest",
    summary="Ingest a knowledge document",
    status_code=status.HTTP_200_OK,
)
async def ingest_endpoint(
    document: UploadFile = File(
        ...,
        description="PDF, DOCX, or image file to ingest into the vector store.",
    ),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> JSONResponse:
    """
    Ingest a knowledge document into MongoDB Atlas Vector Search.

    Accepts PDF, DOCX, or image files. The file is saved temporarily,
    processed, and then always cleaned up.
    """
    # 1. Validate content type
    content_type: str = document.content_type or ""
    if not _is_supported(content_type):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                "Accepted: application/pdf, "
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document, image/*."
            ),
        )

    # 2. Save to temp path
    original_name: str = document.filename or "unknown_file"
    suffix: str = Path(original_name).suffix
    unique_filename: str = f"{uuid.uuid4().hex}{suffix}"
    temp_path: Path = UPLOADS_DIR / unique_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(document.file, buffer)
        logger.info(
            "Saved upload '%s' -> '%s' (%d bytes)",
            original_name,
            temp_path,
            temp_path.stat().st_size,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {exc}",
        ) from exc

    # 3. Ingest + guaranteed cleanup
    try:
        result = pipeline.ingest(
            file_path=str(temp_path),
            original_name=original_name,
            content_type=content_type,
        )
        logger.info(
            "Ingestion complete for '%s': %d chunk(s) stored.",
            original_name,
            result.chunks_stored,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Document ingested successfully.",
                "data": {
                    "chunks_stored": result.chunks_stored,
                    "original_name": result.original_name,
                    "upload_date": result.upload_date,
                },
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
