# backend/agents/answer_agent.py
import os
import time
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
    "Write a direct, self-contained answer — match the length to the question: one short sentence "
    "for a narrow factual question, up to three sentences for a broader one. Do not pad. "
    "If the answer is not in the context, say: 'I couldn't find relevant information in the document.'"
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


class AnswerAgent:
    def __init__(self, api_key: str = None):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        llm = ChatCerebras(
            model="llama3.1-8b",
            temperature=0,
            api_key=api_key or os.getenv("CEREBRAS_API_KEY"),
        )
        self._chain = prompt | llm | StrOutputParser()

    def run(self, query: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        # Retry transient queue/rate errors from Cerebras (shared free tier
        # gets bursty "queue_exceeded" 429s that can take >30s to clear).
        last_err = None
        for attempt in range(8):
            try:
                return self._chain.invoke({"context": context, "question": query})
            except Exception as e:
                msg = str(e)
                # token_quota_exceeded is the daily TPD wall — retrying is pointless.
                if "token_quota_exceeded" in msg:
                    raise
                if "429" in msg or "queue_exceeded" in msg or "503" in msg:
                    last_err = e
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise
        raise last_err
