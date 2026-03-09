import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # --- General ---
    PROJECT_NAME: str = "Enterprise RAG Knowledge Base"
    API_V1_STR: str = "/api/v1"
    
    # --- Database (Metadata) ---
    # using SQLite for local dev, easy to switch to Postgres later
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./local_knowledge_base.db")

    # --- Vector DB (Pinecone) & AI (OpenAI) ---
    # We will populate these in the next phase
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_ENV: str = Field(default="gcp-starter")
    OPENAI_API_KEY: str = Field(default="")

    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)

    # --- Retrieval ---
    # Minimum cosine similarity score to consider a chunk relevant.
    
    SIMILARITY_THRESHOLD: float = Field(default=0.5)
    RERANK_TOP_K: int = Field(default=10)
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()