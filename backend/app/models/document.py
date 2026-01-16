from sqlalchemy import Column, Integer, String, DateTime, Enum
from datetime import datetime
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
    file_url = Column(String) # Path to S3 or local storage
    
    # Store the file hash to prevent duplicate uploads
    file_hash = Column(String, unique=True, index=True)
    
    status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Error message if processing failed
    error_message = Column(String, nullable=True)

    def __repr__(self):
        return f"<Document {self.filename} - {self.status}>"