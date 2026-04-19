# backend/agents/tts_agent.py
import os
import openai


class TTSAgent:
    def __init__(self, audio_dir: str = None):
        self.audio_dir = audio_dir or os.getenv("AUDIO_DIR", "./audio")

    def run(self, text: str, session_id: str, message_id: str) -> str:
        session_dir = os.path.join(self.audio_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)

        file_path = os.path.join(session_dir, f"{message_id}.mp3")

        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
        )

        with open(file_path, "wb") as f:
            f.write(response.content)

        return file_path
