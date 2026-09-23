import os
import uuid
from dotenv import load_dotenv
from langchain_core.tools import tool
import pymupdf4llm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, VectorParams, SparseVectorParams, Distance,
    SparseVector, Prefetch, FusionQuery, Fusion, Modifier
)
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from langchain_text_splitters import MarkdownTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import base64

load_dotenv()

# ============================================================
# Module-level clients — created ONCE, reused everywhere
# ============================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")          # dense, 384-dim
sparse_model = SparseTextEmbedding("Qdrant/bm25")             # sparse, BM25-style

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

vision_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

COLLECTION_NAME = "my_rag_collection"


def ensure_collection_exists(collection_name=COLLECTION_NAME, vector_size=384):
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
            # BM25-style sparse vectors need the IDF modifier, otherwise Qdrant
            # scores them with a plain dot product instead of proper IDF weighting.
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}
        )
        print(f"Created collection: {collection_name}")
    else:
        print(f"Collection '{collection_name}' already exists — skipping creation")


# ============================================================
# STEP 1: Extract text + tables + image metadata
# ============================================================

def extract_pdf_content(pdf_path, image_output_dir="./extracted_images"):
    return pymupdf4llm.to_markdown(
        pdf_path, page_chunks=True, write_images=True,
        image_path=image_output_dir, image_format="png", dpi=150
    )


# ============================================================
# STEP 2: Chunk text/tables
# ============================================================

def _get_page_number(page: dict, fallback: int) -> int:
    """pymupdf4llm's per-page metadata key for the page number has varied
    across versions ('page', 'page_number', or absent). Fall back to the
    page's position in the list (1-indexed) if neither key is present."""
    metadata = page.get("metadata") or {}
    return metadata.get("page") or metadata.get("page_number") or fallback


def chunk_text_and_tables(page_data, chunk_size=500, chunk_overlap=50):
    splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for idx, page in enumerate(page_data):
        page_num = _get_page_number(page, idx + 1)
        for s in splitter.split_text(page["text"]):
            chunks.append({"type": "text", "content": s, "page": page_num})
    return chunks


# ============================================================
# STEP 3: Caption images
# ============================================================

def _get_image_path(img: dict):
    """pymupdf4llm's image dict shape has varied across versions
    ('file', 'path', 'filename') — check all of them."""
    for key in ("file", "path", "filename"):
        val = img.get(key)
        if val:
            return val
    return None


def caption_image(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": "Describe this image/figure/diagram in 2-3 sentences for search indexing."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
    ])
    response = vision_llm.invoke([message])
    return response.content


def process_images(page_data):
    image_chunks = []
    for idx, page in enumerate(page_data):
        page_num = _get_page_number(page, idx + 1)
        for img in page.get("images", []):
            img_path = _get_image_path(img)
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                caption = caption_image(img_path)
            except Exception as exc:
                # A single failed caption shouldn't take down the whole ingest.
                print(f"[warning] Failed to caption image '{img_path}' (page {page_num}): {exc}")
                continue
            image_chunks.append({
                "type": "image", "content": caption,
                "image_path": img_path, "page": page_num
            })
    return image_chunks


# ============================================================
# STEP 4: Embed (dense + sparse) + upsert
# ============================================================

def ingest_pdf(pdf_path, collection_name=COLLECTION_NAME):
    document_id = str(uuid.uuid4())
    document_name = pdf_path.split("/")[-1]

    page_data = extract_pdf_content(pdf_path)
    text_chunks = chunk_text_and_tables(page_data)
    image_chunks = process_images(page_data)
    all_chunks = text_chunks + image_chunks

    if not all_chunks:
        print(f"No content extracted from '{document_name}' — nothing to ingest.")
        return document_id

    # Batch-embed all chunk contents in one shot instead of one call per chunk.
    contents = [c["content"] for c in all_chunks]
    dense_vecs = embedder.encode(contents).tolist()
    sparse_vecs = list(sparse_model.embed(contents))

    points = []
    for chunk, dense_vec, sparse_vec in zip(all_chunks, dense_vecs, sparse_vecs):
        payload = {
            "text": chunk["content"],
            "type": chunk["type"],
            "document_id": document_id,
            "document_name": document_name,
            "page": chunk["page"],
        }
        if chunk["type"] == "image":
            payload["image_path"] = chunk["image_path"]

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_vec,
                "sparse": SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist())
            },
            payload=payload
        ))

    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant_client.upsert(collection_name=collection_name, points=points[i:i+batch_size])

    print(f"Ingested '{document_name}': {len(text_chunks)} text chunks, {len(image_chunks)} image chunks")
    return document_id


def create_collection_and_ingest(pdf_path, collection_name=COLLECTION_NAME):
    """Call this from your UI's upload handler."""
    ensure_collection_exists(collection_name)
    return ingest_pdf(pdf_path, collection_name)


# ============================================================
# Retrieval tool — hybrid search, LLM-callable
# ============================================================

@tool
def retrieve_relevant_chunks(query: str, top_k: int = 5) -> str:
    """
    Search the ingested documents using hybrid retrieval (semantic similarity + keyword matching)
    to find information relevant to the query. Use this whenever the user's question could be
    answered from uploaded PDF documents or reports.
    """
    dense_query = embedder.encode(query).tolist()
    sparse_query = list(sparse_model.embed([query]))[0]

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_query, using="dense", limit=top_k * 4),
            Prefetch(
                query=SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                using="sparse",
                limit=top_k * 4
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    )

    if not results.points:
        return "No relevant information found in the documents."

    formatted = []
    for hit in results.points:
        p = hit.payload
        text = (p.get("text") or "")[:800]
        source = f"[Page {p.get('page', '?')}, {p.get('type', 'text')}, doc: {p.get('document_name', '?')}]"
        formatted.append(f"{source}\n{text}")

    return "\n\n---\n\n".join(formatted)
