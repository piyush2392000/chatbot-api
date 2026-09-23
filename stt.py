"""Speech-to-text helper using Groq's Whisper API.

Kept as its own module for the same reason tts.py is separate — voice
concerns stay out of the chat graph (backend_mcp.py) and the ingestion
pipeline (retrieval.py). Uses Groq directly since GROQ_API_KEY is already
configured for the chat LLM, and Groq's hosted Whisper is fast and has a
generous free tier — no extra API key to manage.
"""

import os

import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_STT_MODEL = "whisper-large-v3-turbo"


def speech_to_text(audio_bytes: bytes, filename: str = "audio.wav", model: str = DEFAULT_STT_MODEL) -> str:
    """Transcribe recorded audio bytes to text. Returns "" on empty input
    or failure rather than raising, so a flaky network call doesn't crash
    the Streamlit script."""
    if not audio_bytes:
        return ""
    if not GROQ_API_KEY:
        print("[warning] GROQ_API_KEY is not set — cannot transcribe audio.")
        return ""

    try:
        response = requests.post(
            GROQ_TRANSCRIBE_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (filename, audio_bytes)},
            data={"model": model},
            timeout=30,
        )
        response.raise_for_status()
        return (response.json().get("text") or "").strip()
    except requests.RequestException as exc:
        print(f"[warning] Groq speech-to-text failed: {exc}")
        return ""
