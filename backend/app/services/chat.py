from app.services.vector_store import vector_service

from app.core.config import settings
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# chat service where user can ask questions and get answers from their uploaded file

class ChatService:
    async def get_answer(self, query: str) -> dict:
        # Retrieve the context found by Pinecone from our PDF text
        context_results = await vector_service.search(query)

        if not context_results:
            return {
                "answer": "No context found to retrieve",
                "sources": []
            }
        
        # creating system prompt where Augmentation will happen

        context_text = "\n\n".join([r['text'] for r in context_results])

        system_prompt = f"""
        You are an expert assistant for an enterprise knowledge base.
        Use the following pieces of context to answer the user's question.
        If the answer is not in the context, just say, "I don't know".
        Don't try to make up ans answer.

        Context: {context_text}
        """

        # Call OpenAI LLM
        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role":"user", "content": query}
            ],
            temperature=0

        )

        # extract answer and sources
        answer = response.choices[0].message.content
        # get unique sources
        sources = list(set([r['source'] for r in context_results])) 

        return {"answer": answer, 'sources': sources}

# store the class in a variable and use it anywhere
chat_service = ChatService()

