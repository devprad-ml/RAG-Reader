from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat import chat_service

router = APIRouter()

@router.post("/query", response_model=ChatResponse)
async def chat_with_docs(request: ChatRequest):
    # receives user query -> finds relevant PDF chunks -> generates answer using LLM

    try:
        response = await chat_service.get_answer(request.query)
        return response
    
    except Exception as e:
        print(f"Error:{e}")
        raise HTTPException(status_code=500, detail=str(e))