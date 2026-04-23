# backend/tests/test_ingestion_agent.py
import json
import os
import pytest
from unittest.mock import MagicMock, patch
from agents.ingestion_agent import IngestPDFAgent


def test_ingestion_creates_collection(sample_pdf_bytes, tmp_path):
    with patch("agents.ingestion_agent.Chroma.from_documents") as mock_chroma, \
         patch("agents.ingestion_agent.HuggingFaceEmbeddings"):
        agent = IngestPDFAgent(chroma_dir=str(tmp_path), chunks_dir=str(tmp_path / "chunks"))
        agent.run(pdf_bytes=sample_pdf_bytes, session_id="test-session-123")
        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args[1]
        assert call_kwargs["collection_name"] == "session_test-session-123"


def test_ingestion_persists_chunks_for_bm25(sample_pdf_bytes, tmp_path):
    chunks_dir = tmp_path / "chunks"
    with patch("agents.ingestion_agent.Chroma.from_documents"), \
         patch("agents.ingestion_agent.HuggingFaceEmbeddings"):
        agent = IngestPDFAgent(chroma_dir=str(tmp_path), chunks_dir=str(chunks_dir))
        agent.run(pdf_bytes=sample_pdf_bytes, session_id="abc")

    chunks_file = chunks_dir / "abc.json"
    assert chunks_file.exists()
    chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
    assert isinstance(chunks, list) and chunks
    assert any("Paris" in c for c in chunks)


def test_ingestion_raises_on_empty_pdf():
    agent = IngestPDFAgent(chroma_dir="/tmp/test_chroma")
    with pytest.raises(ValueError, match="PDF is empty"):
        agent.run(pdf_bytes=b"", session_id="test-session-123")
