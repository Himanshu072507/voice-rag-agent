# backend/tests/test_routes.py
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

@pytest.fixture
def client():
    import sys
    sys.modules.pop('main', None)
    with patch("main.IngestPDFAgent"), \
         patch("main.RetrievalAgent"), \
         patch("main.AnswerAgent"), \
         patch("main.TTSAgent"):
        from main import app
        return TestClient(app)

def test_upload_returns_session_id(client):
    with patch("main.IngestPDFAgent") as mock_ingest_cls:
        mock_ingest_cls.return_value.run = MagicMock()
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) == 36  # UUID format

def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]

def test_chat_returns_answer_and_audio_url(client, tmp_path):
    with patch("main.RetrievalAgent") as mock_ret_cls, \
         patch("main.AnswerAgent") as mock_ans_cls, \
         patch("main.TTSAgent") as mock_tts_cls:
        mock_ret_cls.return_value.run.return_value = ["Paris is the capital."]
        mock_ans_cls.return_value.run.return_value = "Paris."
        import os as _os
        audio_dir = _os.getenv("AUDIO_DIR", "/tmp/test_audio")
        mock_tts_cls.return_value.run.return_value = _os.path.join(audio_dir, "sess", "msg.mp3")

        response = client.post("/chat", json={
            "session_id": "sess-123",
            "query": "What is the capital?",
            "message_id": "msg-001",
        })

    assert response.status_code == 200
    data = response.json()
    assert data["answer_text"] == "Paris."
    assert "audio_url" in data

def test_chat_returns_404_for_missing_session(client):
    with patch("main.RetrievalAgent") as mock_ret_cls:
        mock_ret_cls.return_value.run.side_effect = Exception("Collection not found")
        response = client.post("/chat", json={
            "session_id": "nonexistent",
            "query": "test",
            "message_id": "msg-001",
        })
    assert response.status_code == 404
