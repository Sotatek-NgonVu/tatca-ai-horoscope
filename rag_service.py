"""
rag_service.py
══════════════════════════════════════════════════════════════════════════════
Core ingestion logic for the Tử Vi RAG pipeline.

Responsibilities:
  1. Load documents from a local file path using the correct LangChain loader.
  2. Chunk documents with RecursiveCharacterTextSplitter.
  3. Enrich chunk metadata (original filename + upload timestamp).
  4. Embed chunks with a LOCAL sentence-transformers model (no API key needed)
     and store them in MongoDB Atlas via MongoDBAtlasVectorSearch.

Embeddings model: paraphrase-multilingual-mpnet-base-v2
  • Runs entirely on your machine — zero API calls, zero cost.
  • Downloads once (~500 MB) to ~/.cache/huggingface/ on first run.
  • 768-dimensional vectors; supports Vietnamese and 50+ languages.
  • MongoDB Atlas Vector Search index must be configured with numDimensions=768.

Public API:
  ingest_document(file_path, original_name, content_type) -> dict
  get_vector_store() -> MongoDBAtlasVectorSearch
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MONGODB_URI: str = os.environ["MONGODB_URI"]
DB_NAME: str = "tuvi_ai"
COLLECTION_NAME: str = "knowledge_base"
INDEX_NAME: str = "vector_index"

CHUNK_SIZE: int = 1_000
CHUNK_OVERLAP: int = 200

# Multilingual sentence-transformers model — supports Vietnamese out of the box.
# Downloaded once to ~/.cache/huggingface/ then reused from disk.
EMBEDDING_MODEL: str = "paraphrase-multilingual-mpnet-base-v2"

# ── Module-level singletons ───────────────────────────────────────────────────
# Both are cheap to reuse across requests; initialised once at import time.

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# SentenceTransformerEmbeddings wraps the sentence-transformers library so
# LangChain / MongoDBAtlasVectorSearch can call .embed_documents() on it.
# The model is loaded from local cache after the first download.
_embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

logger.info("Embedding model loaded: %s (768-dim, local)", EMBEDDING_MODEL)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock OCR — replace with Claude Vision call in bot_service.py (already done).
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_ocr(file_path: str) -> list[Document]:
    """
    Placeholder OCR for image files ingested via POST /api/ingest.

    In practice, images sent through the Telegram bot are handled by
    bot_service._ocr_image_with_claude() instead. This placeholder is only
    reached if someone uploads an image directly to the /api/ingest endpoint.
    """
    logger.info("Mock OCR called for image file: %s", file_path)
    dummy_text = (
        "[OCR placeholder] — image ingested via API endpoint.\n"
        f"File: {os.path.basename(file_path)}\n"
        "For real OCR, send the image via the Telegram bot instead."
    )
    return [
        Document(
            page_content=dummy_text,
            metadata={"source": file_path, "loader": "mock_ocr"},
        )
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Document Loader Router
# ═══════════════════════════════════════════════════════════════════════════════

def _load_document(file_path: str, content_type: str) -> list[Document]:
    """
    Route a file to the appropriate LangChain document loader.

    Rules (evaluated in order):
      • PDF   → PyPDFLoader
      • image → _mock_ocr()
      • DOCX  → Docx2txtLoader
      • other → raise ValueError
    """
    logger.info("Loading: path=%s  type=%s", file_path, content_type)

    if content_type == "application/pdf" or file_path.lower().endswith(".pdf"):
        return PyPDFLoader(file_path).load()

    if content_type.startswith("image/"):
        return _mock_ocr(file_path)

    if (
        file_path.lower().endswith(".docx")
        or "wordprocessingml" in content_type
        or content_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        )
    ):
        return Docx2txtLoader(file_path).load()

    raise ValueError(
        f"Unsupported content type '{content_type}'. "
        "Supported: PDF, DOCX, image/*."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Metadata Enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def _enrich_metadata(chunks: list[Document], original_name: str) -> list[Document]:
    """Add original_name and upload_date (UTC ISO-8601) to every chunk."""
    upload_date: str = datetime.now(timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata["original_name"] = original_name
        chunk.metadata["upload_date"] = upload_date
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  Vector Store — shared instance factory
# ═══════════════════════════════════════════════════════════════════════════════

def _get_collection() -> Any:
    """Return the raw PyMongo Collection for the knowledge base."""
    client: MongoClient = MongoClient(MONGODB_URI)
    return client[DB_NAME][COLLECTION_NAME]


def get_vector_store() -> MongoDBAtlasVectorSearch:
    """
    Return a MongoDBAtlasVectorSearch instance ready for similarity search.

    Shared by both:
      - ingest_document()  (writes via from_documents)
      - bot_service.py     (reads via as_retriever)

    The local _embeddings singleton is reused — no re-loading the model.
    """
    return MongoDBAtlasVectorSearch(
        collection=_get_collection(),
        embedding=_embeddings,
        index_name=INDEX_NAME,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Public Ingestion Entry-Point
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_document(
    file_path: str,
    original_name: str,
    content_type: str,
) -> dict[str, Any]:
    """
    Full ingestion pipeline for a single document.

    Steps:
      1. Load  — parse file with the appropriate LangChain loader.
      2. Split — chunk with RecursiveCharacterTextSplitter.
      3. Enrich — add original_name + upload_date to metadata.
      4. Embed & store — convert to 768-dim vectors locally, upsert to MongoDB.

    Returns:
        {
          "status":        "success",
          "chunks_stored": int,
          "original_name": str,
          "upload_date":   str (ISO-8601 UTC),
        }
    """
    logger.info("Starting ingestion for '%s'", original_name)

    # 1. Load
    raw_docs: list[Document] = _load_document(file_path, content_type)
    logger.info("Loaded %d page(s)", len(raw_docs))

    # 2. Split
    chunks: list[Document] = _splitter.split_documents(raw_docs)
    logger.info("Split into %d chunk(s) (size=%d, overlap=%d)",
                len(chunks), CHUNK_SIZE, CHUNK_OVERLAP)

    # 3. Enrich
    upload_date = datetime.now(timezone.utc).isoformat()
    chunks = _enrich_metadata(chunks, original_name)

    # 4. Embed locally + store in MongoDB Atlas
    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=_embeddings,      # ← local sentence-transformers, no API call
        collection=_get_collection(),
        index_name=INDEX_NAME,
    )
    logger.info("Stored %d chunk(s) → %s.%s", len(chunks), DB_NAME, COLLECTION_NAME)

    return {
        "status": "success",
        "chunks_stored": len(chunks),
        "original_name": original_name,
        "upload_date": upload_date,
    }
