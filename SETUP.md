# Setup & Run Guide — Tu Vi RAG Chatbot v4.0

## Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | >= 3.11 | `python3 --version` |
| pip | bundled with Python | `pip3 --version` |
| Docker | any | `docker --version` |
| MongoDB Compass | any | GUI — open the app |
| ngrok (dev only) | any | `ngrok version` |

---

## Step 1 — Create a Virtual Environment

> **Why?** Your system (Debian/Ubuntu) blocks `pip install` system-wide since Python 3.12.
> A virtual environment is the clean, correct solution.

```bash
# Navigate to the project folder
cd ~/tat_ca_ai

# Create the virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

Your terminal prompt will change to show `(.venv)` — this means it's active.

> **Every time you open a new terminal**, run `source .venv/bin/activate` again before running the server.

---

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all packages into `.venv/` only — your system Python is untouched.

> No large model downloads needed — embeddings use the Google Gemini API (free tier).

---

## Step 3 — Configure Environment Variables

```bash
cp .env.example .env
nano .env   # or: code .env / vim .env
```

Fill in the **required** values:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-...       # From https://console.anthropic.com/settings/keys
MONGODB_URI=mongodb://admin:admin@localhost:27018/?authSource=admin&directConnection=true   # Atlas-local container
GOOGLE_API_KEY=your-google-key     # From https://aistudio.google.com/apikey (free)

# Optional (bot won't start without these, but the API still works)
TELEGRAM_BOT_TOKEN=123456:ABC-...  # From @BotFather on Telegram
WEBHOOK_BASE_URL=https://xxxx.ngrok.io   # See Step 5 below
```

All other settings have sensible defaults. See `.env.example` for the full list.

> **Embeddings:** The app uses **`paraphrase-multilingual-mpnet-base-v2`** (sentence-transformers) —
> 768 dimensions, Vietnamese-native, runs **fully locally** — no API key or internet required after
> the first model download.

---

## Step 4 — Get Your Telegram Bot Token

1. Open Telegram, search for **@BotFather**
2. Send `/newbot`
3. Follow the prompts (choose a name and username)
4. Copy the token (format: `123456789:ABC-xxxx`) -> paste into `.env`

---

## Step 5 — Expose Local Server with ngrok (Development)

Telegram requires a **public HTTPS URL** to send webhook updates.
ngrok creates a secure tunnel from the internet to your local machine.

```bash
# In a SEPARATE terminal (keep this running alongside the server)
ngrok http 8000
```

You will see output like:
```
Forwarding  https://a1b2c3d4abcd.ngrok-free.app -> http://localhost:8000
```

Copy the `https://...` URL and set it in your `.env`:
```env
WEBHOOK_BASE_URL=https://a1b2c3d4abcd.ngrok-free.app
```

> The ngrok URL changes every time you restart ngrok (free tier).
> Update `.env` and restart the server when that happens.

---

## Step 6 — Set Up Local MongoDB with Vector Search

### 6a — Start the MongoDB Atlas-Local Container

The project includes a `docker-compose.yaml` that runs **`mongodb/mongodb-atlas-local`** —
a local MongoDB image that supports Atlas Vector Search (the standard `mongo` image does NOT).

```bash
cd ~/tat_ca_ai
docker compose up -d
```

This starts a MongoDB instance on **port 27018** (to avoid conflicts with other MongoDB containers).

Verify it's running:
```bash
docker ps | grep mongodb-tuvi
```

### 6b — Connect with MongoDB Compass

1. Open **MongoDB Compass**
2. Connect to:
   ```
   mongodb://admin:admin@localhost:27018/?authSource=admin
   ```
3. Create database **`tuvi_ai`** with collection **`knowledge_base`** if they don't exist yet

### 6c — Create the Vector Search Index

> **Important:** The embedding model produces **768-dimensional** vectors.
> Use `"numDimensions": 768` — NOT 1536 (that was for OpenAI).

Connect to the container via `mongosh` and create the index:

```bash
docker exec -it mongodb-tuvi mongosh -u admin -p admin --authenticationDatabase admin
```

```javascript
use tuvi_ai

db.knowledge_base.createSearchIndex(
  "vector_index",
  "vectorSearch",
  {
    fields: [
      {
        type: "vector",
        path: "embedding",
        numDimensions: 768,
        similarity: "cosine"
      }
    ]
  }
);
```

Wait ~10 seconds, then verify the index is active:
```javascript
db.knowledge_base.listSearchIndexes().toArray()
```

You should see `"status": "READY"` for `vector_index`.

You can also verify the index in **Compass** under the **Search Indexes** tab of the `knowledge_base` collection.

---

## Step 7 — Start the Server

Make sure your virtual environment is active (`source .venv/bin/activate`), then:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup you should see:
```
INFO  Connecting to MongoDB: db=tuvi_ai
INFO  MongoDB connection verified successfully
INFO  Registering Telegram webhook: https://xxxx.ngrok.io/telegram/webhook
INFO  Telegram webhook registered successfully
INFO  Application startup complete.
```

---

## Step 8 — Ingest Knowledge Documents

Before the bot can give good answers, feed it Tu Vi reference material:

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/api/ingest \
  -F "document=@/path/to/tuvi_book.pdf"

# Ingest a Word document
curl -X POST http://localhost:8000/api/ingest \
  -F "document=@/path/to/tuvi_notes.docx"
```

Successful response:
```json
{
  "status": "success",
  "message": "Document ingested successfully.",
  "data": {
    "chunks_stored": 87,
    "original_name": "tuvi_book.pdf",
    "upload_date": "2026-04-07T10:30:00+00:00"
  }
}
```

---

## Step 9 — Test the Telegram Bot

1. Open Telegram and find your bot (the username you set in @BotFather)
2. Send `/start` -> you should receive the welcome message
3. Send any **horoscope / Tu Vi image** (photo, not file)
4. The bot replies: "Dang phan tich la so cua ban..."
5. After 15-30 seconds, you receive the full Vietnamese Tu Vi interpretation

---

## Step 10 — Run Tests

```bash
# Make sure venv is active
source .venv/bin/activate

# Run all tests
python3 -m pytest tests/ -v

# Run only unit tests
python3 -m pytest tests/unit/ -v

# Run only integration tests
python3 -m pytest tests/integration/ -v
```

Expected: **31 tests pass** (26 unit + 5 integration).

---

## Quick Reference — Daily Workflow

```bash
# Terminal 1 — start MongoDB + ngrok
cd ~/tat_ca_ai
docker compose up -d
ngrok http 8000

# Terminal 2 — activate venv and start server
cd ~/tat_ca_ai
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/ingest` | Upload a PDF/DOCX/image to the knowledge base |
| `POST` | `/telegram/webhook` | Telegram webhook receiver (called by Telegram automatically) |
| `GET` | `/health` | Liveness probe -> `{"status": "ok"}` |
| `GET` | `/docs` | Swagger UI (interactive API docs) |

---

## Project Structure (v3.0)

```
tat_ca_ai/
|-- app/
|   |-- main.py                    # App factory + lifespan
|   |-- config/settings.py         # Pydantic Settings
|   |-- core/
|   |   |-- exceptions.py          # Exception hierarchy
|   |   |-- interfaces.py          # ABC interfaces
|   |-- domain/
|   |   |-- models.py              # Dataclasses
|   |   |-- pipeline.py            # RAG Pipeline orchestrator
|   |-- services/                   # Concrete implementations
|   |   |-- document_loader.py
|   |   |-- embedding.py
|   |   |-- llm.py
|   |   |-- ocr.py
|   |   |-- vector_store.py
|   |-- infrastructure/
|   |   |-- database.py            # MongoDB connection
|   |   |-- dependencies.py        # DI composition root
|   |-- api/routes/                 # REST endpoints
|   |-- bot/telegram/               # Telegram bot handlers
|-- tests/                          # 31 tests (unit + integration)
|-- .env.example
|-- requirements.txt
|-- pyproject.toml
```

See `ARCHITECTURE.md` for detailed design decisions and data flow diagrams.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `externally-managed-environment` error | Use the virtual environment — see Step 1 |
| `Telegram webhook NOT registered` in logs | Check `TELEGRAM_BOT_TOKEN` and `WEBHOOK_BASE_URL` in `.env` |
| Bot doesn't reply | Make sure ngrok is running and `WEBHOOK_BASE_URL` matches the ngrok URL |
| `AuthenticationError` from Anthropic | Check `ANTHROPIC_API_KEY` in `.env` |
| Empty RAG context | No documents ingested yet — run Step 8 first |
| ngrok URL changed | Update `WEBHOOK_BASE_URL` in `.env`, restart the server |
| `(.venv)` missing from prompt | Run `source .venv/bin/activate` again |
| Slow first startup | Normal — first request initializes the Gemini embedding client |
| `Google API key missing` | Set `GOOGLE_API_KEY` in `.env` — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| MongoDB connection refused | Make sure the container is running: `docker compose up -d` |
| `SearchNotEnabled` error | You're using a plain `mongo` image — must use `mongodb/mongodb-atlas-local` (see Step 6) |
| MongoDB index errors | Check `numDimensions` is **768** (not 1536) in your vector index |
| Compass can't connect | Verify container is running and connect to `localhost:27018` (not 27017) |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the venv |
| Tests fail | Run from project root: `python3 -m pytest tests/ -v` |

---

## Changes from v2.0

| Area | v2.0 (Before) | v3.0 (After) |
|---|---|---|
| **Structure** | 3 flat files (`main.py`, `rag_service.py`, `bot_service.py`) | Layered `app/` package with 20+ modules |
| **Config** | Raw `os.getenv()` scattered everywhere | Centralized `Pydantic Settings` with validation |
| **Architecture** | Functions calling functions | Ports & Adapters (interfaces + DI) |
| **Entry point** | `uvicorn main:app` | `uvicorn app.main:app` |
| **Testing** | None | 31 tests (pytest) |
| **Error handling** | Try/except in handlers | Domain exception hierarchy with FastAPI handlers |
| **Document loading** | Inline PDF/DOCX parsing | Extensible loader registry (chain-of-responsibility) |
| **Bot** | Coupled to business logic | Separated into client/handlers/router |
