import hashlib
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.models.document import Document, ProcessingStatus
from app.services.vector_store import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()

    # Compute a SHA-256 fingerprint of the raw bytes.
    # This catches duplicate uploads regardless of filename changes.
    file_hash = hashlib.sha256(content).hexdigest()

    # --- Duplicate check ---
    # The Document model already had a unique file_hash column — wire it up.
    existing = await db.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    existing_doc = existing.scalar_one_or_none()

    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail=f"Document already uploaded as '{existing_doc.filename}' (id={existing_doc.id})."
        )

    # Create the DB record before processing so we have something to update on failure.
    new_doc = Document(
        filename=file.filename,
        file_hash=file_hash,
        status=ProcessingStatus.PROCESSING
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # --- Vector processing ---
    # If this raises, we catch it below, mark the record FAILED with the error
    # message, and re-raise so the client still gets a 500.
    # Without this, failed uploads are left stuck in PROCESSING forever.
    try:
        result = await vector_service.process_pdf(content, file.filename)
        new_doc.status = ProcessingStatus.COMPLETED
        await db.commit()

        return {
            "id": new_doc.id,
            "filename": file.filename,
            "chunks": result["chunks_processed"],
            "message": "File processed and indexed successfully."
        }

    except Exception as e:
        logger.exception("Vector processing failed for '%s'", file.filename)
        new_doc.status = ProcessingStatus.FAILED
        new_doc.error_message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))