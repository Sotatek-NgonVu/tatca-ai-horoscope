# Architecture — Tu Vi RAG Chatbot v3.0

## High-Level Overview

```
User (Telegram)                External Services
     |                              |
     v                              v
+-----------+    +-----------+  +-----------+
| Telegram  |    | FastAPI   |  | Anthropic |
| Webhook   |--->| Ingest    |  | Claude    |
| Router    |    | Endpoint  |  | API       |
+-----------+    +-----------+  +-----------+
     |                |              ^
     v                v              |
+----------------------------------------+
|          RAGPipeline (Domain)          |
|  ingest()        analyze_image()       |
+----------------------------------------+
     |         |         |          |
     v         v         v          v
+--------+ +------+ +--------+ +--------+
|Document| |Vector| |  LLM   | |  OCR   |
|Loaders | |Store | |Service | |Service |
+--------+ +------+ +--------+ +--------+
     |         |
     v         v
+-----------+ +-----------+
| pypdf     | | MongoDB   |
| docx2txt  | | Atlas     |
+-----------+ | Local     |
              | Vector    |
              | Search    |
              +-----------+
```

---

## Project Structure

```
tat_ca_ai/
|-- app/
|   |-- main.py                          # FastAPI app factory + lifespan
|   |-- config/
|   |   |-- settings.py                  # Pydantic Settings (typed, validated)
|   |-- core/
|   |   |-- exceptions.py               # Domain exception hierarchy
|   |   |-- interfaces.py               # ABC interfaces (ports)
|   |-- domain/
|   |   |-- models.py                   # Dataclasses: Chunk, IngestionResult, AnalysisResult
|   |   |-- pipeline.py                 # RAGPipeline orchestrator
|   |-- services/                        # Concrete implementations (adapters)
|   |   |-- document_loader.py          # PDF, DOCX, Image loaders + registry
|   |   |-- embedding.py               # SentenceTransformerEmbeddingService
|   |   |-- llm.py                      # ClaudeLLMService (with extended thinking)
|   |   |-- ocr.py                      # ClaudeOCRService (Claude Vision)
|   |   |-- vector_store.py            # MongoVectorStore (Atlas Vector Search)
|   |-- infrastructure/
|   |   |-- database.py                 # MongoDB connection manager
|   |   |-- dependencies.py            # FastAPI DI wiring (composition root)
|   |-- api/
|   |   |-- routes/
|   |       |-- health.py              # GET /health
|   |       |-- ingest.py              # POST /api/ingest
|   |-- bot/
|       |-- base.py                     # Abstract BaseBotHandler
|       |-- telegram/
|           |-- client.py              # Telegram HTTP client
|           |-- handlers.py            # Message handlers (start, text, photo)
|           |-- router.py             # Webhook router + dispatcher
|-- tests/
|   |-- conftest.py                     # Shared fixtures + mock services
|   |-- unit/
|   |   |-- test_pipeline.py           # RAGPipeline unit tests
|   |   |-- test_document_loader.py    # Loader + registry tests
|   |   |-- test_vector_store.py       # Mock vector store tests
|   |-- integration/
|       |-- test_api.py                # FastAPI endpoint tests
|-- .env.example                        # Environment variable template
|-- requirements.txt                    # Python dependencies
|-- pyproject.toml                      # Project metadata + tool config
```

---

## Architectural Principles

### 1. Ports & Adapters (Hexagonal Architecture)

The **core** layer defines abstract interfaces (ports). The **services** layer provides concrete implementations (adapters). The **domain** layer depends only on abstractions.

```
interfaces.py (ports)         services/ (adapters)
---------------------         --------------------
EmbeddingService       <---   SentenceTransformerEmbeddingService
LLMService             <---   ClaudeLLMService
OCRService             <---   ClaudeOCRService
VectorStoreRepository  <---   MongoVectorStore
DocumentLoader         <---   PDFDocumentLoader, DocxDocumentLoader, ImageDocumentLoader
```

### 2. Dependency Injection via FastAPI

All service wiring happens in `infrastructure/dependencies.py` (the composition root). Route handlers declare dependencies via `Depends()`. No global state or module-level singletons in business logic.

### 3. Single Responsibility per Module

| Module | Single Responsibility |
|---|---|
| `settings.py` | Environment variable parsing and validation |
| `pipeline.py` | Orchestrate the RAG workflow (no I/O knowledge) |
| `database.py` | MongoDB connection lifecycle |
| `dependencies.py` | Service instantiation and wiring |
| `router.py` | HTTP routing only (no business logic) |
| `handlers.py` | Message handling logic (no HTTP knowledge) |

---

## Data Flow

### Pipeline 1 — Document Ingestion

```
POST /api/ingest (file upload)
  |
  v
ingest_router  ->  RAGPipeline.ingest()
                      |
                      +--> DocumentLoaderRegistry.load()
                      |       |-- PDFDocumentLoader (pypdf)
                      |       |-- DocxDocumentLoader (docx2txt)
                      |       |-- ImageDocumentLoader (placeholder)
                      |
                      +--> RecursiveCharacterTextSplitter.split()
                      |
                      +--> Enrich metadata (original_name, upload_date)
                      |
                      +--> MongoVectorStore.add_documents()
                              |-- SentenceTransformerEmbeddingService.embed()
                              |-- MongoDB insert_many()
```

### Pipeline 2 — Image Analysis (Telegram)

```
Telegram sends photo -> POST /telegram/webhook
  |
  v
telegram_router  ->  "Dang phan tich..." (immediate ack)
  |
  v
BackgroundTask:  handle_photo_message()
  |
  +--> TelegramClient.download_file()
  |
  +--> RAGPipeline.analyze_image()
          |
          +--> ClaudeOCRService.extract_text()     [Claude Vision]
          |
          +--> MongoVectorStore.similarity_search()  [RAG retrieval]
          |
          +--> ClaudeLLMService.generate()           [Final answer]
  |
  +--> TelegramClient.send_message()  ->  User receives interpretation
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Local embeddings** (sentence-transformers) | No API cost, no rate limits, works offline after first download |
| **Claude for both OCR and LLM** | One API key, consistent quality, Vision API handles complex charts |
| **Extended thinking** in ClaudeLLMService | Better reasoning for complex Tu Vi interpretation |
| **Chain-of-responsibility** for document loaders | Easy to add new formats without modifying existing code |
| **Background tasks** for photo processing | Telegram requires webhook response within 5 seconds |
| **No OpenAI dependency** | Fully Anthropic-based stack, simpler to manage |
| **768-dim vectors** (multilingual mpnet) | Native Vietnamese support, good balance of quality vs. size |

---

## Exception Hierarchy

```
AppError (base)
|-- DocumentLoadError
|   |-- UnsupportedDocumentTypeError
|-- EmbeddingError
|-- VectorStoreError
|-- OCRError
|-- LLMError
    |-- LLMAuthenticationError
    |-- LLMRateLimitError
```

All exceptions carry a `message` field. The pipeline catches and converts them to user-friendly responses (never leaks stack traces to end users).

---

## Configuration

All configuration flows through `app/config/settings.py` (Pydantic Settings). Values are loaded from `.env` with typed defaults:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *required* | Claude API key |
| `MONGODB_URI` | *required* | MongoDB connection string (Atlas-local on port 27018) |
| `TELEGRAM_BOT_TOKEN` | `""` | Telegram bot token (optional) |
| `WEBHOOK_BASE_URL` | `""` | Public HTTPS URL for webhook (optional) |
| `MONGODB_DB_NAME` | `tuvi_ai` | Database name |
| `MONGODB_COLLECTION_NAME` | `knowledge_base` | Collection name |
| `MONGODB_INDEX_NAME` | `vector_index` | Vector search index name |
| `CLAUDE_MODEL` | `claude-opus-4-6` | Claude model for LLM + OCR |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-mpnet-base-v2` | Local embedding model |
| `RAG_TOP_K` | `5` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `1000` | Text splitter chunk size |
| `CHUNK_OVERLAP` | `200` | Text splitter overlap |

---

## Testing Strategy

| Layer | Approach |
|---|---|
| **Unit tests** | Mock all external dependencies via `conftest.py` mock services |
| **Integration tests** | FastAPI `TestClient` with dependency overrides (no real DB/API) |
| **Test count** | 31 tests (26 unit + 5 integration) |

Run: `python3 -m pytest tests/ -v`
