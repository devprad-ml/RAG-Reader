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

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()