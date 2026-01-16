from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str

    filter_document_id: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]