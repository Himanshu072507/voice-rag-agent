import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RetrievalAgent:
    def __init__(self, chroma_dir: str = None):
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "./chroma_db")

    def run(self, query: str, session_id: str, k: int = 5) -> list[str]:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
            persist_directory=self.chroma_dir,
        )
        docs = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
