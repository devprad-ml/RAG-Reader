from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 1. Create the Async Engine
# check_same_thread=False is needed only for SQLite
engine = create_async_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=True, # Set to False in production
)

# 2. Create the Session Factory
# This is what we will use to interact with the DB in our API endpoints
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# 3. Base Class for Models
Base = declarative_base()

# 4. Dependency for FastAPI Endpoints
async def get_db():
    """
    Dependency function that yields a database session.
    Used in FastAPI endpoints like: def my_endpoint(db: AsyncSession = Depends(get_db))
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()