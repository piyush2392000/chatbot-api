"""FastAPI wrapper around the existing LangGraph chatbot, RAG ingestion,
and voice helpers — for non-Streamlit clients (mobile app, etc.).

Deliberately thin: all real logic stays in backend_mcp.py, retrieval.py,
Tts.py, and stt.py. This file only adds HTTP endpoints on top of them, so
the Streamlit app and this API stay in sync automatically (same graph,
same tools, same checkpointer — no duplicated logic to drift out of sync).

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Requires POSTGRES_URI to be set (see backend_mcp.py) when running this
alongside the Streamlit app, since two processes must not share the local
SQLite fallback file concurrently.
"""

import os
import tempfile
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend_mcp import chatbot, get_thread_state, retrieve_all_threads, run_async
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from retrieval import create_collection_and_ingest
from Tts import AVAILABLE_VOICES, DEFAULT_VOICE, text_to_speech
from stt import speech_to_text

app = FastAPI(title="LangGraph MCP Chatbot API")

# Adjust for your actual mobile app's origin(s) in production instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")  # set this in your Space secrets to require a key


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Minimal shared-secret auth. Skipped entirely if API_KEY isn't set
    (e.g. local dev) — set API_KEY in production so this isn't a fully
    open endpoint on the internet."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/threads")
def list_threads(_: None = Depends(require_api_key)):
    return {"thread_ids": [str(t) for t in retrieve_all_threads()]}


@app.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str, _: None = Depends(require_api_key)):
    state = get_thread_state(thread_id)
    messages = state.values.get("messages", [])
    return {
        "thread_id": thread_id,
        "messages": [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
        ],
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(require_api_key)):
    """Non-streaming turn: send a message, get the final reply back.
    Good default for mobile clients that just want a single response."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "run_name": "chat_turn"}

    result = run_async(
        chatbot.ainvoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    )
    final_message = result["messages"][-1]
    return ChatResponse(thread_id=thread_id, reply=final_message.content)


@app.get("/voices")
def list_voices():
    """So the frontend doesn't need to hardcode/duplicate this list."""
    return {"voices": AVAILABLE_VOICES, "default": DEFAULT_VOICE}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, _: None = Depends(require_api_key)):
    """Streaming turn as Server-Sent Events. Emits three event types:
    - "thread_id": the thread this turn is running on (once, at the start)
    - "tool": a tool the agent just invoked, e.g. "retrieve_relevant_chunks"
    - "message" (default/unlabeled): a chunk of the assistant's reply text
    - "done": end of stream
    Content is split across multiple `data:` lines per the SSE spec so
    multi-line replies survive the round trip intact."""
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "run_name": "chat_turn"}

    def _sse(data: str, event: Optional[str] = None) -> str:
        lines = "\n".join(f"data: {line}" for line in data.split("\n"))
        prefix = f"event: {event}\n" if event else ""
        return f"{prefix}{lines}\n\n"

    async def event_generator():
        yield _sse(thread_id, event="thread_id")
        async for message_chunk, _metadata in chatbot.astream(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                yield _sse(tool_name, event="tool")
            elif isinstance(message_chunk, AIMessage) and message_chunk.content:
                yield _sse(message_chunk.content)
        yield _sse("{}", event="done")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest")
def ingest_pdf_endpoint(file: UploadFile = File(...), _: None = Depends(require_api_key)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        document_id = create_collection_and_ingest(tmp_path)
        return {"document_id": document_id, "filename": file.filename}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE


@app.post("/tts")
def tts_endpoint(req: TTSRequest, _: None = Depends(require_api_key)):
    audio_bytes = text_to_speech(req.text, req.voice)
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="Speech synthesis failed.")
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/stt")
def stt_endpoint(file: UploadFile = File(...), _: None = Depends(require_api_key)):
    audio_bytes = file.file.read()
    text = speech_to_text(audio_bytes, filename=file.filename or "audio.wav")
    if not text:
        raise HTTPException(status_code=502, detail="Transcription failed or produced empty text.")
    return {"text": text}
