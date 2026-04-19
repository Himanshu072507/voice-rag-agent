# backend/tests/conftest.py
import os
import pytest

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setenv("AUDIO_DIR", "/tmp/test_audio")
    monkeypatch.setenv("CHROMA_DIR", "/tmp/test_chroma")
    os.makedirs("/tmp/test_audio", exist_ok=True)

@pytest.fixture
def sample_pdf_bytes():
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The capital of France is Paris.")
    return doc.tobytes()
