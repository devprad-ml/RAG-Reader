import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.db.session import engine, Base
from app.api.endpoints import routes, chat
# --- Lifecycle Events ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events.
    Created tables automatically for simplicity in this dev phase.
    """
    # Startup: Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("LOG: Database tables created.")
    
    yield
    
    # Shutdown (if needed)
    print("LOG: Shutting down...")

# --- App Definition ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)
# CORS config
# allows the frontend at port 3000 to access backend at 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"], # allows all CRUD operations
    allow_headers=["*"]
    )

    
# --- Basic Health Check ---
@app.get("/")
async def root():
    return {
        "message": "Enterprise RAG API is running", 
        "docs_url": "/docs"
    }
app.include_router(routes.router, prefix="/api/v1/routes", tags=["Documents"])  # upload and chunking of docs
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])  # getting the embeddings and answering the questions



# --- Runner ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)