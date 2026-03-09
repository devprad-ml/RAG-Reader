import logging
import os
from pypdf import PdfReader
from io import BytesIO
from typing import List
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from app.core.config import settings

logger = logging.getLogger(__name__)

# init clients
# using AsyncOpenAI for non-blocking calls
aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# init pinecone vector database
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

# Cross-encoder reranker — loaded once at startup, runs locally (no API cost).
# It reads the query and each candidate chunk *together*, making it much more
# accurate than cosine similarity alone. MiniLM is fast and small enough for prod.
reranker = CrossEncoder(settings.RERANKER_MODEL)

INDEX_NAME = "rag-knowledge-base"

class VectorStoreService:
    # pipeline to populate vector database
    async def process_pdf(self, file_content: bytes, filename: str) -> dict:
        # extract text
        text = self._extract_text_from_pdf(file_content)
        
        # divide the text into semantic chunks with overlap
        chunks = self._chunk_text(text)

        await self._embed_and_store(chunks, filename)

        return {"chunks_processed": len(chunks), "status" :"success"}
    
    async def search(self, query: str, limit: int = 3) -> List[dict]:
        """
        Three-stage retrieval pipeline:
          1. HyDE — generate a hypothetical answer and embed it instead of the raw query.
          2. Vector retrieval — cast a wide net with Pinecone using that embedding.
          3. Reranking — precision pass with a cross-encoder on the candidates.

        Why HyDE? A user query like "what were Q3 revenues?" sits in a different
        part of embedding space than the document chunk that answers it. A hypothetical
        answer ("Q3 revenues were $X...") lives much closer to real answer chunks,
        so retrieval recall improves significantly — especially for factual questions.
        """
        index = pc.Index(INDEX_NAME)

        # --- Stage 1: HyDE — embed a hypothetical answer, not the raw query ---
        embedding_input = await self._hypothetical_document(query)

        query_embedding_response = await aclient.embeddings.create(
            input=embedding_input,
            model="text-embedding-3-small"
        )
        query_vector = query_embedding_response.data[0].embedding

        # --- Stage 2: Vector retrieval (cast a wide net) ---
        search_results = index.query(
            vector=query_vector,
            top_k=settings.RERANK_TOP_K,
            include_metadata=True
        )

        # Filter by minimum similarity threshold before passing to reranker
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

        # --- Stage 3: Cross-encoder reranking (precision pass) ---
        # Rerank against the original query, not the hypothetical answer —
        # we want chunks relevant to what the user actually asked.
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:limit]

    async def _hypothetical_document(self, query: str) -> str:
        """
        HyDE: generate a short passage that *would* answer the query,
        then use that passage's embedding for retrieval instead of the query itself.

        The embedding of a fluent answer aligns much more closely with real document
        chunks than the embedding of a short question. Falls back to the raw query
        if the LLM call fails, so retrieval always continues.
        """
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
            # Non-fatal — fall back to raw query so search always proceeds
            logger.warning("HyDE generation failed for query '%s', falling back to raw query", query)
            return query
    
    # helper function to extract text from pdf

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    # helper to divide text into semantic chunks with overlap
    def _chunk_text(self, text: str) -> List[str]:
        """
        Uses RecursiveCharacterTextSplitter to split on natural boundaries
        (paragraphs → sentences → words) before falling back to characters.
        chunk_overlap ensures context is never lost at chunk edges.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    
    # embed using embeddings model and store
    async def _embed_and_store(self, chunks: List[str], filename: str):
        existing_index = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in existing_index:
            pc.create_index(
                name=INDEX_NAME,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        index = pc.Index(INDEX_NAME)
        # generate embeddings in batch
        

        response = await aclient.embeddings.create(
            input=chunks,
            model='text-embedding-3-small'
        )

        # prepare vectors for pinecone

        vectors_to_upsert = []
        for i, (chunk, embedding_data) in enumerate(zip(chunks, response.data)):
            vector_id = f"{filename}_{i}"
            metadata = {
                "text": chunk, 
                "source": filename,
                "chunk_index": i
            }
            vectors_to_upsert.append((vector_id, embedding_data.embedding, metadata))

        index.upsert(vectors=vectors_to_upsert)

vector_service = VectorStoreService()