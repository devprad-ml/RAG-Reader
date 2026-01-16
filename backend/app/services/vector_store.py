import os
from pypdf import PdfReader
from io import BytesIO
from typing import List
from openai import AsyncOpenAI
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

# init clients 
# using AsyncOpenAI for non-blocking calls
aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# init pinecone vector database
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

INDEX_NAME = "rag-knowledge-base"

class VectorStoreService:
    # pipeline to populate vector database
    async def process_pdf(self, file_content: bytes, filename: str) -> dict:
        # extract text
        text = self._extract_text_from_pdf(file_content)
        
        # divide the text into chunks for better context retention
        chunks = self._chunk_text(text, chunk_size =500)

        await self._embed_and_store(chunks, filename)

        return {"chunks_processed": len(chunks), "status" :"success"}
    
    async def search(self, query: str, limit: int=3,) -> List[dict]:
        # we will convert the query to vectors and search for similar vectors
        index = pc.Index(INDEX_NAME)

        query_embedding_response = await aclient.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_vector = query_embedding_response.data[0].embedding

        # search our vector database
        search_results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True
        )

        # format the results for output
        contexts=[]
        if 'matches' in search_results:
            for match in search_results['matches']:
                if match['score'] > 0.5:
                    contexts.append({
                        "text": match['metadata']['text'],
                        "source": match['metadata']['source'],
                        "score": match['score']

                    })
        return contexts
    
    # helper function to extract text from pdf

    def _extract_text_from_pdf(self, file_content: bytes) -> str:
        reader = PdfReader(BytesIO(file_content))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    # helper to divide text into chunks 

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
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
