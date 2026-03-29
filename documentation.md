# Enterprise RAG Knowledge Base — Documentation

## Overview

A full-stack Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about their content. The system uses a three-stage retrieval pipeline (HyDE, vector search, cross-encoder reranking) to find the most relevant document chunks, then generates answers with an LLM while maintaining conversation history.

Designed to handle very large documents (2000+ pages) through background processing with real-time progress tracking.

---

## Architecture

```
┌─────────────────┐        HTTP        ┌──────────────────────────────────────────┐
│   Next.js UI    │ ◄──────────────► │           FastAPI Backend                │
│   (port 3000)   │                    │           (port 8000)                    │
└─────────────────┘                    │                                          │
                                       │  ┌──────────┐  ┌────────────────────┐   │
                                       │  │ SQLite   │  │  OpenAI API        │   │
                                       │  │ (metadata)│  │  (embeddings+LLM) │   │
                                       │  └──────────┘  └────────────────────┘   │
                                       │                                          │
                                       │  ┌──────────────┐  ┌───────────────┐    │
                                       │  │ Pinecone     │  │ CrossEncoder  │    │
                                       │  │ (vector DB)  │  │ (local model) │    │
                                       │  └──────────────┘  └───────────────┘    │
                                       └──────────────────────────────────────────┘
```

---

## Project Structure

```
├── backend/
│   ├── main.py                          # FastAPI app, lifespan, CORS, router mounts
│   ├── requirements.txt                 # Python dependencies
│   ├── .env                             # Environment variables (not committed)
│   └── app/
│       ├── core/
│       │   └── config.py                # Pydantic settings from .env
│       ├── db/
│       │   └── session.py               # SQLAlchemy async engine & session
│       ├── models/
│       │   └── document.py              # Document ORM model with progress tracking
│       ├── schemas/
│       │   └── chat.py                  # Pydantic request/response models
│       ├── services/
│       │   ├── vector_store.py          # PDF processing, embedding, HyDE, reranking
│       │   └── chat.py                  # RAG answer generation with conversation memory
│       └── api/
│           └── endpoints/
│               ├── routes.py            # POST /upload, GET /status/{id}
│               └── chat.py              # POST /query — question answering
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx               # Root layout
│       │   ├── page.tsx                 # Main chat + upload UI with progress polling
│       │   └── globals.css              # Tailwind imports
│       └── lib/
│           └── api.ts                   # Axios client for backend
```

---

## Tech Stack

| Layer        | Technology                                       |
|------------- |--------------------------------------------------|
| Frontend     | Next.js 16, React 19, Tailwind CSS 4, TypeScript |
| Backend      | FastAPI, Uvicorn, Python 3.10+                   |
| Database     | SQLite (via SQLAlchemy + aiosqlite)               |
| Vector DB    | Pinecone Serverless (AWS us-east-1, cosine)       |
| Embeddings   | OpenAI `text-embedding-3-small` (1536 dims)       |
| LLM          | OpenAI `gpt-4o-mini`                              |
| Reranker     | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)    |
| Text Split   | LangChain `RecursiveCharacterTextSplitter`         |
| Tokenizer    | `tiktoken` (cl100k_base) for history truncation    |

---

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenAI API key
- Pinecone API key

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
PROJECT_NAME="Enterprise RAG"
DATABASE_URL="sqlite+aiosqlite:///./local_knowledge_base.db"
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
PINECONE_ENV="gcp-starter"
CHUNK_SIZE=512
CHUNK_OVERLAP=64
SIMILARITY_THRESHOLD=0.5
RERANK_TOP_K=10
RERANKER_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_BATCH_SIZE=256
PAGE_BATCH_SIZE=50
MAX_HISTORY_TOKENS=4000
```

Start the server:

```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI is available at `http://localhost:3000`. The backend runs at `http://localhost:8000` (Swagger docs at `/docs`).

---

## API Endpoints

### `GET /`

Health check. Returns `{ "message": "Enterprise RAG API is running", "docs_url": "/docs" }`.

### `POST /api/v1/routes/upload`

Upload a PDF document. Processing happens in the background — the response returns immediately.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` (PDF only)
- **Response (200):**
  ```json
  {
    "id": 1,
    "filename": "mahabharata.pdf",
    "status": "processing",
    "message": "Upload received. Processing started in the background."
  }
  ```
- **Errors:** `400` (non-PDF), `409` (duplicate file hash)

### `GET /api/v1/routes/status/{doc_id}`

Poll processing progress for a document.

- **Response (200):**
  ```json
  {
    "id": 1,
    "filename": "mahabharata.pdf",
    "status": "processing",
    "total_chunks": 0,
    "processed_chunks": 142,
    "error_message": null
  }
  ```
  `status` is one of: `pending`, `processing`, `completed`, `failed`.

### `POST /api/v1/chat/query`

Ask a question about uploaded documents.

- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "query": "What is the refund policy?",
    "history": [
      { "role": "user", "content": "Tell me about returns" },
      { "role": "assistant", "content": "The return policy states..." }
    ]
  }
  ```
  `history` is optional (defaults to `[]`). Including it enables multi-turn conversations where the LLM understands references like "tell me more about that." History is automatically truncated to stay within `MAX_HISTORY_TOKENS`.

- **Response (200):**
  ```json
  {
    "answer": "According to the handbook, the refund policy is...",
    "sources": ["handbook.pdf"]
  }
  ```

---

## Data Flow

### Document Upload Pipeline

```
PDF file
  → SHA-256 hash (duplicate check against DB)
  → HTTP response returns immediately (background task starts)
  → Extract text in page batches (PAGE_BATCH_SIZE pages at a time)
  → Chunk each batch (RecursiveCharacterTextSplitter, 512 chars, 64 overlap)
  → Embed chunks in batches (EMBEDDING_BATCH_SIZE, with retry + backoff)
  → Upsert vectors to Pinecone (batches of 100)
  → Update progress in DB (frontend polls for updates)
  → Mark document COMPLETED or FAILED
```

### Query Pipeline (Three-Stage Retrieval)

```
User query
  → Stage 1: HyDE — LLM generates a hypothetical answer
  → Embed the hypothetical answer (not the raw query)
  → Stage 2: Vector search — Pinecone top-K retrieval (K=10)
  → Filter by similarity threshold (>0.5)
  → Stage 3: Cross-encoder reranking — rerank against original query
  → Top 3 chunks become context
  → Truncate conversation history to fit token budget
  → LLM generates final answer with context + history
```

**Why HyDE?** A user query like "what were Q3 revenues?" sits in a different part of embedding space than the document chunk that answers it. A hypothetical answer ("Q3 revenues were $X...") lives much closer to real answer chunks, so retrieval recall improves — especially for factual questions.

**Why reranking?** Cosine similarity is fast but shallow. The cross-encoder reads the query and each candidate chunk together as a pair, enabling much deeper semantic matching. This two-pass approach (fast recall → precise rerank) balances speed and accuracy.

---

## Database Model

**`documents` table:**

| Column             | Type                  | Notes                                |
|------------------- |---------------------- |--------------------------------------|
| `id`               | Integer (PK)          | Auto-increment                       |
| `filename`         | String (indexed)      | Original upload name                 |
| `file_url`         | String                | Storage path (S3 or local)           |
| `file_hash`        | String (unique, idx)  | SHA-256 of file bytes                |
| `status`           | Enum                  | PENDING / PROCESSING / COMPLETED / FAILED |
| `created_at`       | DateTime (tz-aware)   | UTC timestamp                        |
| `total_chunks`     | Integer               | Total chunks after processing completes |
| `processed_chunks` | Integer               | Chunks processed so far (for progress) |
| `error_message`    | String (nullable)     | Populated on FAILED status           |

---

## Configuration Reference

All settings are read from `backend/.env` via Pydantic:

| Variable               | Default                                      | Description                                      |
|----------------------- |----------------------------------------------|--------------------------------------------------|
| `PROJECT_NAME`         | `Enterprise RAG Knowledge Base`              | App title                                        |
| `DATABASE_URL`         | `sqlite+aiosqlite:///./local_knowledge_base.db` | SQLAlchemy connection string                  |
| `OPENAI_API_KEY`       | —                                            | OpenAI API key (required)                        |
| `PINECONE_API_KEY`     | —                                            | Pinecone API key (required)                      |
| `PINECONE_ENV`         | `gcp-starter`                                | Pinecone environment                             |
| `CHUNK_SIZE`           | `512`                                        | Characters per text chunk                        |
| `CHUNK_OVERLAP`        | `64`                                         | Overlap between chunks                           |
| `SIMILARITY_THRESHOLD` | `0.5`                                        | Minimum cosine similarity for retrieval          |
| `RERANK_TOP_K`         | `10`                                         | Candidates retrieved before reranking            |
| `RERANKER_MODEL`       | `cross-encoder/ms-marco-MiniLM-L-6-v2`      | Local cross-encoder model                        |
| `EMBEDDING_BATCH_SIZE` | `256`                                        | Chunks per OpenAI embedding API call             |
| `PAGE_BATCH_SIZE`      | `50`                                         | PDF pages processed per iteration                |
| `MAX_HISTORY_TOKENS`   | `4000`                                       | Max tokens for conversation history in LLM prompt |

---

## Key Design Decisions

1. **Background processing for large documents.** Upload returns immediately and processing runs as an async background task. The frontend polls a status endpoint for real-time progress. This prevents HTTP timeouts on large PDFs (2000+ pages).

2. **Page-batch extraction.** Instead of loading all text from a large PDF into memory at once, pages are extracted and processed in configurable batches (`PAGE_BATCH_SIZE`). This keeps memory bounded regardless of document size.

3. **Embedding retry with backoff.** OpenAI embedding calls use exponential backoff (up to 5 retries) to handle rate limits and transient errors gracefully during large batch operations.

4. **Token-aware history truncation.** Conversation history is truncated from the oldest messages using `tiktoken` token counting, keeping the most recent context within `MAX_HISTORY_TOKENS` to prevent context window overflow.

5. **History goes to LLM, not vector search.** Sending full conversation history to the embedding model would dilute the query signal and hurt retrieval recall. History is only included in the LLM prompt for conversational context.

6. **Cross-encoder loads once at startup** (`vector_store.py` module level). This avoids re-downloading and re-initializing the model on every query.

7. **Duplicate detection via file hash.** SHA-256 of the raw PDF bytes catches re-uploads regardless of filename changes, preventing wasted embedding costs and duplicate vectors.

8. **HyDE falls back gracefully.** If the hypothetical document generation fails, the system falls back to embedding the raw query so search always proceeds.

9. **User-facing error messages.** Both upload and query errors surface backend error details to the user instead of failing silently. Network errors show a specific diagnostic message.
