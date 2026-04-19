# backend/agents/answer_agent.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say: 'I couldn't find relevant information in the document.'"
)

USER_PROMPT = "Context:\n{context}\n\nQuestion: {question}"


class AnswerAgent:
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._chain = prompt | llm | StrOutputParser()

    def run(self, query: str, chunks: list[str]) -> str:
        context = "\n\n".join(chunks)
        return self._chain.invoke({"context": context, "question": query})
