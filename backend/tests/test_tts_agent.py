# backend/tests/test_tts_agent.py
import os
import pytest
from unittest.mock import MagicMock, patch
from agents.tts_agent import TTSAgent

def test_tts_agent_saves_mp3(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"fake_mp3_bytes"

    with patch("agents.tts_agent.openai.audio.speech.create", return_value=mock_response):
        agent = TTSAgent(audio_dir=str(tmp_path))
        file_path = agent.run(text="Hello world.", session_id="sess-1", message_id="msg-1")

    assert file_path.endswith(".mp3")
    assert os.path.exists(file_path)
    with open(file_path, "rb") as f:
        assert f.read() == b"fake_mp3_bytes"

def test_tts_agent_uses_correct_path_structure(tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"bytes"

    with patch("agents.tts_agent.openai.audio.speech.create", return_value=mock_response):
        agent = TTSAgent(audio_dir=str(tmp_path))
        file_path = agent.run(text="Hi", session_id="sess-abc", message_id="msg-xyz")

    assert "sess-abc" in file_path
    assert "msg-xyz" in file_path
