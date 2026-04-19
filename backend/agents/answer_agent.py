# backend/agents/answer_agent.py
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say: 'I couldn't find relevant information in the document.'"
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


class AnswerAgent:
    def __init__(self, api_key: str = None):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=api_key or os.getenv("GROQ_API_KEY"),
        )
        self._chain = prompt | llm | StrOutputParser()

    def run(self, query: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        return self._chain.invoke({"context": context, "question": query})
