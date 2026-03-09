import os
from pypdf import PdfReader
from io import BytesIO
from typing import List
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from app.core.config import settings

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
        Two-stage retrieval pipeline:
          1. Retrieve a broad candidate set from Pinecone using vector similarity.
          2. Rerank candidates with a cross-encoder for precise relevance scoring.

        Why two stages? Embedding similarity is fast but coarse — it compares
        vectors independently. The cross-encoder reads the query and each chunk
        *together*, catching nuance that cosine similarity misses. We fetch more
        than we need (RERANK_TOP_K) then cut down to `limit` after reranking.
        """
        index = pc.Index(INDEX_NAME)

        # --- Stage 1: Vector retrieval (cast a wide net) ---
        query_embedding_response = await aclient.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = query_embedding_response.data[0].embedding

        search_results = index.query(
            vector=query_vector,
            top_k=settings.RERANK_TOP_K,   # fetch more candidates than we'll return
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

        # --- Stage 2: Cross-encoder reranking (precision pass) ---
        # Feed [query, chunk] pairs to the cross-encoder; it outputs a relevance score
        # for each pair jointly — much more accurate than independent vector distances.
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs)

        # Attach rerank scores and sort descending
        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:limit]
    
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