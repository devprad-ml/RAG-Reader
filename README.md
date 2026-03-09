
# RAG Reader

A production-grade Retrieval-Augmented Generation (RAG) system for querying PDF documents using natural language. Upload documents, ask questions, and get accurate, grounded answers with source citations.

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│   Frontend  │────▶│                   FastAPI Backend                 │
│  React + TW │     │                                                    │
└─────────────┘     │  ┌─────────────┐        ┌─────────────────────┐  │
                    │  │  /upload    │        │      /query         │  │
                    │  │             │        │                     │  │
                    │  │ 1. Hash PDF │        │ 1. HyDE expansion   │  │
                    │  │ 2. Dedup    │        │ 2. Vector retrieval │  │
                    │  │ 3. Chunk    │        │ 3. Cross-encoder    │  │
                    │  │ 4. Embed    │        │    rerank           │  │
                    │  │ 5. Store    │        │ 4. LLM generation   │  │
                    │  └──────┬──────┘        └──────────┬──────────┘  │
                    └─────────┼─────────────────────────┼─────────────┘
                              │                         │
                    ┌─────────▼──────┐       ┌──────────▼──────────┐
                    │    Pinecone    │       │       OpenAI         │
                    │  Vector Store  │       │  Embeddings + GPT    │
                    └────────────────┘       └─────────────────────┘
                              │
                    ┌─────────▼──────┐
                    │  SQLite (dev)  │
                    │  Document DB   │
                    └────────────────┘
```

---

## Retrieval Pipeline

Each query goes through three stages designed to maximise both recall and precision:

**Stage 1 — HyDE (Hypothetical Document Embeddings)**
The raw user query is a poor match for document chunks in embedding space. Instead, GPT-4o-mini generates a short hypothetical answer, which is then embedded. Answer embeddings live much closer to real document chunks than question embeddings do.

**Stage 2 — Vector Retrieval**
The hypothetical embedding is used to fetch the top-K candidates from Pinecone via cosine similarity. A wider net is cast here (configurable via `RERANK_TOP_K`) since the reranker will cut it down.

**Stage 3 — Cross-Encoder Reranking**
A `sentence-transformers` cross-encoder reads each `[query, chunk]` pair jointly and produces a precise relevance score. Unlike embedding similarity, this catches nuance like negation and specificity. The top 3 reranked chunks are passed to the LLM.

---

## Features

- **PDF ingestion** — upload PDFs via REST API; text is extracted, chunked, embedded, and indexed automatically
- **Semantic chunking** — `RecursiveCharacterTextSplitter` splits on paragraph and sentence boundaries with configurable overlap, never mid-word
- **Duplicate detection** — SHA-256 fingerprinting prevents re-processing the same file regardless of filename
- **Conversation memory** — stateless multi-turn chat; the client accumulates history and sends it each turn for horizontal scalability
- **Grounded answers** — LLM is constrained to context only; responds "I don't know" rather than hallucinating
- **Source citations** — every answer includes the source filenames it drew from

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11+ |
| Vector DB | Pinecone (Serverless) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Metadata DB | SQLite (dev) / PostgreSQL (prod) |
| Frontend | React, Tailwind CSS, Axios |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [OpenAI API key](https://platform.openai.com)
- [Pinecone API key](https://www.pinecone.io)

### Backend Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rag-reader.git
cd rag-reader/backend

# 2. Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
```

Edit `.env` with your credentials:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk-...
```

```bash
# 5. Start the server
python main.py
```

API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend Setup

```bash
cd ../frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:3000`.

---

## API Reference

### Upload a Document

```http
POST /api/v1/routes/upload
Content-Type: multipart/form-data

file: <PDF file>
```

**Response**
```json
{
  "id": 1,
  "filename": "report.pdf",
  "chunks": 42,
  "message": "File processed and indexed successfully."
}
```

**Error — duplicate file**
```json
{
  "detail": "Document already uploaded as 'report.pdf' (id=1)."
}
```

### Query Documents

```http
POST /api/v1/chat/query
Content-Type: application/json

{
  "query": "What were the Q3 revenues?",
  "history": [
    { "role": "user", "content": "Summarise the report." },
    { "role": "assistant", "content": "The report covers..." }
  ]
}
```

**Response**
```json
{
  "answer": "Q3 revenues totalled $4.2B according to report.pdf, representing a 12% increase year-over-year.",
  "sources": ["report.pdf"]
}
```

---

## Configuration

All tunable parameters are in `backend/app/core/config.py` and can be overridden via environment variables.

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `512` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `64` | Characters shared between adjacent chunks |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity to include a candidate |
| `RERANK_TOP_K` | `10` | Candidates fetched from Pinecone before reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking |

---

## Project Structure

```
rag-reader/
├── backend/
│   ├── main.py                        # FastAPI app entry point
│   ├── requirements.txt
│   └── app/
│       ├── api/
│       │   └── endpoints/
│       │       ├── routes.py          # /upload endpoint
│       │       └── chat.py            # /query endpoint
│       ├── core/
│       │   └── config.py             # Settings and env vars
│       ├── db/
│       │   └── session.py            # SQLAlchemy async session
│       ├── models/
│       │   └── document.py           # Document DB model
│       ├── schemas/
│       │   └── chat.py               # Pydantic request/response schemas
│       └── services/
│           ├── vector_store.py       # Chunking, embedding, HyDE, reranking
│           └── chat.py               # LLM call with conversation history
└── frontend/
    └── ...
```

---

## Roadmap

- [ ] Streaming LLM responses
- [ ] Parallel HyDE + direct query embedding
- [ ] Query embedding cache (Redis)
- [ ] Support for `.docx` and `.txt` file types
- [ ] Per-document filtering in queries
- [ ] PostgreSQL support for production deployments
- [ ] Docker Compose setup

---

## License

MIT
