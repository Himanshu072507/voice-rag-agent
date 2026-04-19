# backend/agents/ingestion_agent.py
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class IngestPDFAgent:
    def __init__(self, chroma_dir: str = None):
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "./chroma_db")

    def run(self, pdf_bytes: bytes, session_id: str) -> None:
        if not pdf_bytes:
            raise ValueError("PDF is empty")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()

        if not raw_text.strip():
            raise ValueError("PDF is empty")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_text(raw_text)
        documents = [
            Document(page_content=chunk, metadata={"session_id": session_id})
            for chunk in chunks
        ]

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=f"session_{session_id}",
            persist_directory=self.chroma_dir,
        )
