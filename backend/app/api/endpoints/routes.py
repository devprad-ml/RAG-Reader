from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.models.document import Document, ProcessingStatus
from app.services.vector_store import vector_service

#init router
router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # read file contents
        content = await file.read()

        # save file to SQLite (local DB)
        new_doc = Document(
            filename=file.filename,
            status=ProcessingStatus.PROCESSING
        )

        db.add(new_doc)
        await db.commit()
        await db.refresh(new_doc)

        # now trigger vector processing.(async operation)
        # a real traffic heavy app will have a cache(Celery/ Redis)

        result = await vector_service.process_pdf(content, file.filename)

        # update status when processing is completed

        new_doc.status = ProcessingStatus.COMPLETED
        await db.commit()

        return {
            "id": new_doc.id,
            "filename": file.filename,
            "chunks": result["chunks_processed"],
            "message": "File processed and indexed successfully."
        }
    except Exception as e:
        # log error and update DB
        print(f"Error: {e}")

        raise HTTPException(status_code=500, detail=str(e))
    
        
    


