import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import generator, parsers
from app.chunker import chunk_pages
from app.config import settings
from app.embeddings import get_embedder
from app.store import VectorStore

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = state.get("embedder") or get_embedder()
    state["embedder"] = embedder
    state["store"] = state.get("store") or VectorStore(settings.data_dir, embedder.dim)
    yield


app = FastAPI(
    title="RAG Document Query System",
    description="Upload PDFs, Word files, text or scanned images and ask questions answered from their content.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)
    doc_ids: list[str] | None = None


class Source(BaseModel):
    ref: int
    filename: str
    page: int
    score: float
    ocr: bool
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list[Source]
    retrieval_ms: float
    total_ms: float


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    store: VectorStore = state["store"]
    return {
        "status": "ok",
        "documents": len(store.documents()),
        "chunks": store.index.ntotal,
        "llm": bool(settings.openai_api_key),
    }


@app.post("/documents", status_code=201)
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        pages = parsers.parse(file.filename, data)
    except parsers.UnsupportedFileType as e:
        raise HTTPException(415, str(e))
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "No text could be extracted from this file")
    vectors = state["embedder"].encode([c.text for c in chunks])
    doc_id = state["store"].add_document(file.filename, chunks, vectors)
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(pages),
        "ocr_pages": sum(p.ocr for p in pages),
        "chunks": len(chunks),
    }


@app.get("/documents")
def list_documents():
    return state["store"].documents()


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    removed = state["store"].delete_document(doc_id)
    if not removed:
        raise HTTPException(404, "Document not found")
    return {"doc_id": doc_id, "chunks_removed": removed}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()
    embedder = state["embedder"]
    hits = state["store"].search(embedder.encode([req.question])[0], req.top_k, req.doc_ids)
    hits = [h for h in hits if h.score >= settings.min_score]
    t1 = time.perf_counter()
    answer, mode = generator.generate(req.question, hits, embedder)
    t2 = time.perf_counter()
    return QueryResponse(
        question=req.question,
        answer=answer,
        mode=mode,
        sources=[
            Source(
                ref=i,
                filename=h.chunk.filename,
                page=h.chunk.page,
                score=round(h.score, 4),
                ocr=h.chunk.ocr,
                excerpt=h.chunk.text[:400],
            )
            for i, h in enumerate(hits, start=1)
        ],
        retrieval_ms=round((t1 - t0) * 1000, 1),
        total_ms=round((t2 - t0) * 1000, 1),
    )
