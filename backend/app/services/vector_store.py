import asyncio
import logging
from pypdf import PdfReader
from io import BytesIO
from typing import List, Optional, Callable, Awaitable
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from app.core.config import settings

logger = logging.getLogger(__name__)

# init clients
aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Cross-encoder reranker — loaded once at startup, runs locally (no API cost).
reranker = CrossEncoder(settings.RERANKER_MODEL)

INDEX_NAME = "rag-knowledge-base"

# Type alias for the progress callback used during large-document processing.
ProgressCallback = Callable[[int, int], Awaitable[None]]


class VectorStoreService:

    async def process_pdf(
        self,
        file_content: bytes,
        filename: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        """
        Process a PDF in page-batches to keep memory bounded.
        For a 2200-page document this avoids holding the entire extracted
        text + all chunks + all embeddings in memory simultaneously.
        """
        reader = PdfReader(BytesIO(file_content))
        total_pages = len(reader.pages)
        page_batch_size = settings.PAGE_BATCH_SIZE

        self._ensure_index()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        global_chunk_index = 0
        total_chunks_processed = 0

        for page_start in range(0, total_pages, page_batch_size):
            page_end = min(page_start + page_batch_size, total_pages)

            # Extract text for this batch of pages only
            batch_text = ""
            for page in reader.pages[page_start:page_end]:
                batch_text += page.extract_text() or ""

            if not batch_text.strip():
                continue

            chunks = splitter.split_text(batch_text)
            if not chunks:
                continue

            await self._embed_and_store(chunks, filename, start_index=global_chunk_index)

            global_chunk_index += len(chunks)
            total_chunks_processed += len(chunks)

            if on_progress:
                await on_progress(total_chunks_processed, page_end)

            logger.info(
                "Processed pages %d-%d of %d (%d chunks so far) for '%s'",
                page_start + 1, page_end, total_pages, total_chunks_processed, filename
            )

        return {"chunks_processed": total_chunks_processed, "status": "success"}

    async def search(self, query: str, limit: int = 3) -> List[dict]:
        """
        Three-stage retrieval pipeline:
          1. HyDE — generate a hypothetical answer and embed it instead of the raw query.
          2. Vector retrieval — cast a wide net with Pinecone using that embedding.
          3. Reranking — precision pass with a cross-encoder on the candidates.
        """
        index = pc.Index(INDEX_NAME)

        # --- Stage 1: HyDE ---
        embedding_input = await self._hypothetical_document(query)

        query_embedding_response = await aclient.embeddings.create(
            input=embedding_input,
            model="text-embedding-3-small"
        )
        query_vector = query_embedding_response.data[0].embedding

        # --- Stage 2: Vector retrieval ---
        search_results = index.query(
            vector=query_vector,
            top_k=settings.RERANK_TOP_K,
            include_metadata=True
        )

        candidates = []
        if "matches" in search_results:
            for match in search_results["matches"]:
                if match["score"] > settings.SIMILARITY_THRESHOLD:
                    candidates.append({
                        "text": match["metadata"]["text"],
                        "source": match["metadata"]["source"],
                        "vector_score": match["score"],
                    })

        if not candidates:
            return []

        # --- Stage 3: Cross-encoder reranking ---
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:limit]

    async def _hypothetical_document(self, query: str) -> str:
        try:
            response = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write a concise, factual passage (2-3 sentences) that would "
                        f"directly answer the following question. Do not mention that "
                        f"you are generating a hypothetical answer.\n\nQuestion: {query}"
                    )
                }],
                max_tokens=150,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception:
            logger.warning("HyDE generation failed for query '%s', falling back to raw query", query)
            return query

    def _ensure_index(self):
        """Create the Pinecone index if it doesn't exist yet."""
        existing = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in existing:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

    async def _embed_and_store(
        self, chunks: List[str], filename: str, start_index: int = 0
    ):
        """Embed chunks in batches with retry, then upsert to Pinecone."""
        index = pc.Index(INDEX_NAME)
        batch_size = settings.EMBEDDING_BATCH_SIZE
        vectors_to_upsert = []

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            embeddings = await self._embed_with_retry(batch)

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                global_index = start_index + batch_start + i
                vector_id = f"{filename}_{global_index}"
                metadata = {
                    "text": chunk,
                    "source": filename,
                    "chunk_index": global_index
                }
                vectors_to_upsert.append((vector_id, embedding, metadata))

        # Pinecone recommends upserts of ~100 vectors at a time
        for upsert_start in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[upsert_start:upsert_start + 100]
            index.upsert(vectors=batch)

    async def _embed_with_retry(
        self, texts: List[str], max_retries: int = 5
    ) -> List[List[float]]:
        """Call OpenAI embeddings with exponential backoff for rate limits."""
        for attempt in range(max_retries):
            try:
                response = await aclient.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                return [d.embedding for d in response.data]
            except Exception as e:
                error_str = str(e).lower()
                is_retryable = "rate" in error_str or "429" in error_str or "timeout" in error_str
                if is_retryable and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Embedding API error (attempt %d/%d), retrying in %ds: %s",
                                   attempt + 1, max_retries, wait, e)
                    await asyncio.sleep(wait)
                else:
                    raise


vector_service = VectorStoreService()
