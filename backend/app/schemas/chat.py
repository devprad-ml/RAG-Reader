from pydantic import BaseModel
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    """ mirrors the OpenAI message format
    so it can be forwarded to the LLM without any transformation"""
    role: Literal["user", "assistant"] 
    content: str

class ChatRequest(BaseModel):
    query: str

    history: List[ChatMessage] = []

    filter_document_id: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]