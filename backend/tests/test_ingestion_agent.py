# backend/tests/test_ingestion_agent.py
import pytest
from unittest.mock import MagicMock, patch
from agents.ingestion_agent import IngestPDFAgent

def test_ingestion_creates_collection(sample_pdf_bytes, tmp_path):
    mock_vectorstore = MagicMock()
    with patch("agents.ingestion_agent.Chroma.from_documents", return_value=mock_vectorstore) as mock_chroma, \
         patch("agents.ingestion_agent.OpenAIEmbeddings") as mock_embeddings:
        agent = IngestPDFAgent(chroma_dir=str(tmp_path))
        agent.run(pdf_bytes=sample_pdf_bytes, session_id="test-session-123")
        mock_chroma.assert_called_once()
        call_kwargs = mock_chroma.call_args[1]
        assert call_kwargs["collection_name"] == "session_test-session-123"

def test_ingestion_raises_on_empty_pdf():
    agent = IngestPDFAgent(chroma_dir="/tmp/test_chroma")
    with pytest.raises(ValueError, match="PDF is empty"):
        agent.run(pdf_bytes=b"", session_id="test-session-123")
