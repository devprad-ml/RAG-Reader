import asyncio
import hashlib
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db, AsyncSessionLocal
from app.models.document import Document, ProcessingStatus
from app.services.vector_store import vector_service

logger = logging.getLogger(__name__)
router = APIRouter()


async def _process_in_background(doc_id: int, content: bytes, filename: str):
    """
    Runs PDF processing outside the request lifecycle so the upload
    endpoint can return immediately — critical for large documents
    that would otherwise cause an HTTP timeout.
    """
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, doc_id)

        async def on_progress(chunks_done: int, pages_done: int):
            doc.processed_chunks = chunks_done
            await db.commit()

        try:
            result = await vector_service.process_pdf(
                content, filename, on_progress=on_progress
            )
            doc.status = ProcessingStatus.COMPLETED
            doc.total_chunks = result["chunks_processed"]
            doc.processed_chunks = result["chunks_processed"]
            await db.commit()
            logger.info("Background processing completed for '%s': %d chunks", filename, result["chunks_processed"])

        except Exception as e:
            logger.exception("Background processing failed for '%s'", filename)
            doc.status = ProcessingStatus.FAILED
            doc.error_message = str(e)
            await db.commit()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    content = await file.read()

    file_hash = hashlib.sha256(content).hexdigest()

    # --- Duplicate check ---
    existing = await db.execute(
        select(Document).where(Document.file_hash == file_hash)
    )
    existing_doc = existing.scalar_one_or_none()

    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail=f"Document already uploaded as '{existing_doc.filename}' (id={existing_doc.id})."
        )

    new_doc = Document(
        filename=file.filename,
        file_hash=file_hash,
        status=ProcessingStatus.PROCESSING
    )
    db.add(new_doc)
    await db.commit()
    await db.refresh(new_doc)

    # Launch processing as a background task so the HTTP response
    # returns immediately. The frontend polls /status/{id} for progress.
    asyncio.create_task(_process_in_background(new_doc.id, content, file.filename))

    return {
        "id": new_doc.id,
        "filename": file.filename,
        "status": "processing",
        "message": "Upload received. Processing started in the background."
    }


@router.get("/status/{doc_id}")
async def get_document_status(
    doc_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Poll this endpoint to track processing progress for large documents."""
    doc = await db.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status.value,
        "total_chunks": doc.total_chunks,
        "processed_chunks": doc.processed_chunks,
        "error_message": doc.error_message,
    }
