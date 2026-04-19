# backend/main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.ingestion_agent import IngestPDFAgent
from agents.retrieval_agent import RetrievalAgent
from agents.answer_agent import AnswerAgent
from agents.tts_agent import TTSAgent

load_dotenv()

app = FastAPI()

AUDIO_DIR = os.getenv("AUDIO_DIR", "./audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


class ChatRequest(BaseModel):
    session_id: str
    query: str
    message_id: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 10_000_000:
        raise HTTPException(status_code=400, detail="File exceeds 10MB limit")

    session_id = str(uuid.uuid4())

    try:
        agent = IngestPDFAgent()
        agent.run(pdf_bytes=pdf_bytes, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"session_id": session_id}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        retrieval_agent = RetrievalAgent()
        chunks = retrieval_agent.run(query=request.query, session_id=request.session_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a PDF first.")

    answer_agent = AnswerAgent()
    answer_text = answer_agent.run(query=request.query, chunks=chunks)

    tts_agent = TTSAgent()
    try:
        file_path = tts_agent.run(
            text=answer_text,
            session_id=request.session_id,
            message_id=request.message_id,
        )
        relative_path = os.path.relpath(file_path, AUDIO_DIR)
        audio_url = f"/audio/{relative_path}"
    except Exception:
        audio_url = None

    return {"answer_text": answer_text, "audio_url": audio_url}


@app.get("/health")
def health():
    return {"status": "ok"}
