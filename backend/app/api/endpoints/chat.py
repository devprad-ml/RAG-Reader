import logging

from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import chat_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/query", response_model=ChatResponse)
async def chat_with_docs(request: ChatRequest):
    # receives user query -> finds relevant PDF chunks -> generates answer using LLM

    try:
        response = await chat_service.get_answer(
            query=request.query,
            history=request.history)
        return response
    
    except Exception as e:
        logger.exception("Error in /query endpoint")
        raise HTTPException(status_code=500, detail=str(e))