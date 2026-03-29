# Enterprise RAG Knowledge Base — Documentation

## Overview

A full-stack Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about their content. The system uses a three-stage retrieval pipeline (HyDE, vector search, cross-encoder reranking) to find the most relevant document chunks, then generates answers with an LLM while maintaining conversation history.

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
│       │   └── document.py              # Document ORM model with processing status
│       ├── schemas/
│       │   └── chat.py                  # Pydantic request/response models
│       ├── services/
│       │   ├── vector_store.py          # PDF processing, embedding, HyDE, reranking
│       │   └── chat.py                  # RAG answer generation with conversation memory
│       └── api/
│           └── endpoints/
│               ├── routes.py            # POST /upload — PDF upload & indexing
│               └── chat.py              # POST /query — question answering
├── frontend/
│   ├── package.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx               # Root layout
│       │   ├── page.tsx                 # Main chat + upload UI
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

Upload and index a PDF document.

- **Content-Type:** `multipart/form-data`
- **Body:** `file` (PDF only)
- **Response (200):**
  ```json
  {
    "id": 1,
    "filename": "handbook.pdf",
    "chunks": 42,
    "message": "File processed and indexed successfully."
  }
  ```
- **Errors:** `400` (non-PDF), `409` (duplicate file hash), `500` (processing failure)

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
  `history` is optional (defaults to `[]`). Including it enables multi-turn conversations where the LLM understands references like "tell me more about that."

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
  → Extract text (PyPDF)
  → Chunk text (RecursiveCharacterTextSplitter, 512 chars, 64 overlap)
  → Embed chunks (OpenAI text-embedding-3-small)
  → Upsert vectors to Pinecone
  → Update document status in SQLite (COMPLETED / FAILED)
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
  → LLM generates final answer with conversation history
```

**Why HyDE?** A user query like "what were Q3 revenues?" sits in a different part of embedding space than the document chunk that answers it. A hypothetical answer ("Q3 revenues were $X...") lives much closer to real answer chunks, so retrieval recall improves — especially for factual questions.

**Why reranking?** Cosine similarity is fast but shallow. The cross-encoder reads the query and each candidate chunk together as a pair, enabling much deeper semantic matching. This two-pass approach (fast recall → precise rerank) balances speed and accuracy.

---

## Database Model

**`documents` table:**

| Column         | Type                  | Notes                                |
|--------------- |---------------------- |--------------------------------------|
| `id`           | Integer (PK)          | Auto-increment                       |
| `filename`     | String (indexed)      | Original upload name                 |
| `file_url`     | String                | Storage path (S3 or local)           |
| `file_hash`    | String (unique, idx)  | SHA-256 of file bytes                |
| `status`       | Enum                  | PENDING / PROCESSING / COMPLETED / FAILED |
| `created_at`   | DateTime (tz-aware)   | UTC timestamp                        |
| `error_message`| String (nullable)     | Populated on FAILED status           |

---

## Configuration Reference

All settings are read from `backend/.env` via Pydantic:

| Variable               | Default                                      | Description                          |
|----------------------- |----------------------------------------------|--------------------------------------|
| `PROJECT_NAME`         | `Enterprise RAG Knowledge Base`              | App title                            |
| `DATABASE_URL`         | `sqlite+aiosqlite:///./local_knowledge_base.db` | SQLAlchemy connection string      |
| `OPENAI_API_KEY`       | —                                            | OpenAI API key (required)            |
| `PINECONE_API_KEY`     | —                                            | Pinecone API key (required)          |
| `PINECONE_ENV`         | `gcp-starter`                                | Pinecone environment                 |
| `CHUNK_SIZE`           | `512`                                        | Characters per text chunk            |
| `CHUNK_OVERLAP`        | `64`                                         | Overlap between chunks               |
| `SIMILARITY_THRESHOLD` | `0.5`                                        | Minimum cosine similarity for retrieval |
| `RERANK_TOP_K`         | `10`                                         | Candidates retrieved before reranking |
| `RERANKER_MODEL`       | `cross-encoder/ms-marco-MiniLM-L-6-v2`      | Local cross-encoder model            |

---

## Key Design Decisions

1. **History goes to LLM, not vector search.** Sending full conversation history to the embedding model would dilute the query signal and hurt retrieval recall. History is only included in the LLM prompt for conversational context.

2. **Cross-encoder loads once at startup** (`vector_store.py` module level). This avoids re-downloading and re-initializing the model on every query.

3. **Duplicate detection via file hash.** SHA-256 of the raw PDF bytes catches re-uploads regardless of filename changes, preventing wasted embedding costs and duplicate vectors.

4. **Failed uploads are recorded, not lost.** The document status is set to `FAILED` with an error message rather than leaving records stuck in `PROCESSING`.

5. **HyDE falls back gracefully.** If the hypothetical document generation fails, the system falls back to embedding the raw query so search always proceeds.
