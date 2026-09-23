# syntax=docker/dockerfile:1
# API service — LangGraph agent, RAG retrieval, and voice pipeline.
# All the heavy ML deps (torch, transformers, sentence-transformers,
# fastembed) live here now; the frontend no longer needs any of this.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the embedding models into the image at BUILD time, not runtime —
# every container from this image already has the weights on disk, no
# network call to Hugging Face Hub when the API actually starts.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"

# Only the files the API service actually imports — not streamlit_frontend.py
COPY api.py backend_mcp.py retrieval.py Tts.py stt.py ./

ENV HF_HUB_OFFLINE=1

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
