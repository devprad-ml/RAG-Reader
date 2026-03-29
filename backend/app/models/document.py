from sqlalchemy import Column, Integer, String, DateTime, Enum
from datetime import datetime, timezone
import enum
from app.db.session import Base

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(Base):
    """
    SQLAlchemy model representing a file uploaded to the system.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_url = Column(String)  # Path to S3 or local storage

    # SHA-256 fingerprint of the file bytes — prevents duplicate uploads
    # regardless of whether the filename was changed between attempts.
    file_hash = Column(String, unique=True, index=True)

    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    # utcnow() is deprecated in Python 3.12+; use timezone-aware datetime instead.
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Progress tracking for large documents
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)

    # Populated on FAILED status so operators can diagnose without digging through logs.
    error_message = Column(String, nullable=True)

    def __repr__(self):
        return f"<Document {self.filename} - {self.status}>"