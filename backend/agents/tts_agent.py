# backend/agents/tts_agent.py
# TTS is handled by the browser's Web Speech API — this agent is intentionally disabled.

class TTSAgent:
    def run(self, text: str, session_id: str, message_id: str) -> str:
        raise RuntimeError("TTS disabled: using browser speech synthesis")
