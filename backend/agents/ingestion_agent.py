# backend/agents/ingestion_agent.py
import os
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document


class IngestPDFAgent:
    def __init__(self, chroma_dir: str = None, chunks_dir: str = None):
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "./chroma_db")
        self.chunks_dir = chunks_dir or os.getenv("CHUNKS_DIR", "./chunks_store")

    def run(self, pdf_bytes: bytes, session_id: str) -> None:
        if not pdf_bytes:
            raise ValueError("PDF is empty")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.is_encrypted:
                if not doc.authenticate(""):
                    doc.close()
                    raise ValueError("PDF is password-protected and cannot be read")
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Could not parse PDF: {e}") from e
        raw_text = ""
        for page in doc:
            raw_text += page.get_text()
        doc.close()

        if not raw_text.strip():
            raise ValueError("PDF contains no extractable text (may be image-only)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        chunks = splitter.split_text(raw_text)
        documents = [
            Document(page_content=chunk, metadata={"session_id": session_id})
            for chunk in chunks
        ]

        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if hf_token:
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=f"session_{session_id}",
            persist_directory=self.chroma_dir,
        )

        os.makedirs(self.chunks_dir, exist_ok=True)
        chunks_path = os.path.join(self.chunks_dir, f"{session_id}.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
