# Enterprise RAG Knowledge Base

A full-stack RAG (Retrieval-Augmented Generation) system for uploading PDF documents and asking natural-language questions about their content. Built to handle large documents (2000+ pages) with background processing, real-time progress tracking, and a three-stage retrieval pipeline.

## Features

- **Large document support** — processes 2000+ page PDFs in the background with live progress updates
- **Three-stage retrieval** — HyDE (hypothetical document embeddings) + vector search + cross-encoder reranking
- **Multi-turn conversations** — token-aware history truncation keeps context without exceeding model limits
- **Duplicate detection** — SHA-256 hashing prevents re-uploading the same document
- **Error resilience** — embedding retry with exponential backoff, graceful HyDE fallback, user-facing error messages

## Tech Stack

| Layer      | Technology                                        |
|----------- |---------------------------------------------------|
| Frontend   | Next.js 16, React 19, Tailwind CSS 4, TypeScript  |
| Backend    | FastAPI, Python 3.10+                              |
| Database   | SQLite (SQLAlchemy + aiosqlite)                    |
| Vector DB  | Pinecone Serverless                                |
| Embeddings | OpenAI `text-embedding-3-small`                    |
| LLM        | OpenAI `gpt-4o-mini`                               |
| Reranker   | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)     |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Pinecone API key](https://www.pinecone.io/)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
```

Start the server:

```bash
python main.py
```

Backend runs at `http://localhost:8000` (Swagger docs at `/docs`).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI runs at `http://localhost:3000`.

## How It Works

### Upload Pipeline

```
PDF upload → SHA-256 duplicate check → respond immediately
  → background task: extract pages in batches of 50
  → chunk text (512 chars, 64 overlap)
  → embed in batches via OpenAI (with retry + backoff)
  → upsert to Pinecone
  → frontend polls /status/{id} for live progress
```

### Query Pipeline

```
User question
  → HyDE: generate hypothetical answer, embed that instead of raw query
  → Pinecone vector search (top 10 candidates)
  → filter by similarity threshold (>0.5)
  → cross-encoder reranking → top 3 chunks
  → LLM generates answer with conversation history
```

## API

| Method | Endpoint                        | Description                  |
|------- |-------------------------------- |------------------------------|
| GET    | `/`                             | Health check                 |
| POST   | `/api/v1/routes/upload`         | Upload a PDF                 |
| GET    | `/api/v1/routes/status/{id}`    | Poll processing progress     |
| POST   | `/api/v1/chat/query`            | Ask a question               |

## Configuration

All settings via `backend/.env`:

| Variable               | Default  | Description                              |
|----------------------- |----------|------------------------------------------|
| `OPENAI_API_KEY`       | —        | Required                                 |
| `PINECONE_API_KEY`     | —        | Required                                 |
| `CHUNK_SIZE`           | `512`    | Characters per text chunk                |
| `CHUNK_OVERLAP`        | `64`     | Overlap between chunks                   |
| `SIMILARITY_THRESHOLD` | `0.5`    | Minimum cosine similarity for retrieval  |
| `RERANK_TOP_K`         | `10`     | Candidates before reranking              |
| `EMBEDDING_BATCH_SIZE` | `256`    | Chunks per embedding API call            |
| `PAGE_BATCH_SIZE`      | `50`     | PDF pages processed per batch            |
| `MAX_HISTORY_TOKENS`   | `4000`   | Token budget for conversation history    |

See [documentation.md](documentation.md) for full architecture details, database schema, and design decisions.

## Project Structure

```
backend/
  main.py                           # FastAPI app entry point
  app/
    api/endpoints/
      routes.py                     # Upload + status endpoints
      chat.py                       # Query endpoint
    services/
      vector_store.py               # PDF processing, HyDE, reranking
      chat.py                       # RAG answer generation
    core/config.py                  # Settings from .env
    models/document.py              # Document ORM model
    schemas/chat.py                 # Request/response schemas
    db/session.py                   # Database setup

frontend/
  src/
    app/page.tsx                    # Chat UI with upload + progress
    lib/api.ts                      # Backend API client
```

## License

MIT
