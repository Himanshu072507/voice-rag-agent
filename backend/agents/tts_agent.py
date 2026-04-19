# backend/agents/tts_agent.py
import os
import re
import openai


def _safe_path_component(value: str, name: str) -> str:
    if not re.match(r'^[\w\-]+$', value):
        raise ValueError(f"Invalid {name}: must contain only alphanumeric, dash, or underscore characters")
    return value


class TTSAgent:
    def __init__(self, audio_dir: str = None):
        self.audio_dir = audio_dir or os.getenv("AUDIO_DIR", "./audio")

    def run(self, text: str, session_id: str, message_id: str) -> str:
        session_id = _safe_path_component(session_id, "session_id")
        message_id = _safe_path_component(message_id, "message_id")

        session_dir = os.path.join(self.audio_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        file_path = os.path.join(session_dir, f"{message_id}.mp3")

        try:
            response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {e}") from e

        try:
            with open(file_path, "wb") as f:
                f.write(response.content)
        except OSError as e:
            raise RuntimeError(f"Failed to save audio file: {e}") from e

        return file_path
