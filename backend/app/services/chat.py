from typing import List

from app.services.vector_store import vector_service
from app.schemas.chat import ChatMessage
from app.core.config import settings
from openai import AsyncOpenAI

aclient = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


class ChatService:
    async def get_answer(self, query: str, history: List[ChatMessage] = []) -> dict:
        """
        Retrieval-augmented generation with conversation memory.

        Context retrieval uses the current query only — passing the full history
        to the vector search would dilute the embedding and hurt recall.
        History is only forwarded to the LLM, where it belongs.
        """
        context_results = await vector_service.search(query)

        if not context_results:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base.",
                "sources": []
            }

        # Build context block from reranked chunks.
        # Include rerank_score so the LLM can use confidence implicitly.
        context_text = "\n\n".join([
            f"[Source: {r['source']}]\n{r['text']}" for r in context_results
        ])

        system_prompt = f"""You are an expert assistant for an enterprise knowledge base.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't know" — do not speculate.
Be concise and cite the source filename when referencing specific information.

Context:
{context_text}"""

        # Build the message list: system prompt → history → current query.
        # This lets the LLM understand references like "what did you just say about X?"
        messages = [{"role": "system", "content": system_prompt}]

        for turn in history:
            messages.append({"role": turn.role, "content": turn.content})

        messages.append({"role": "user", "content": query})

        response = await aclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        answer = response.choices[0].message.content
        sources = list(set([r["source"] for r in context_results]))

        return {"answer": answer, "sources": sources}


chat_service = ChatService()