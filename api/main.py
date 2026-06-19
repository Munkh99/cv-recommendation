"""FastAPI search backend. Streamlit (or any client) POSTs to /search."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_settings
from src.schema import SearchRequest, SearchResponse
from src.rag import get_rag_client
from src.ingest import ingest_pdf
from src.search import search
from src.tracing import setup_tracing, flush


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_tracing()  # init Langfuse before any agent/LLM call is served
    yield
    flush()  # export buffered traces on shutdown


app = FastAPI(title="CV Recommendation API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    # candidates = records / 4 sections
    n = get_rag_client().store.count()
    return {"records": n, "candidates": n // 4}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    return search(req)


@app.post("/ingest")
def ingest_endpoint() -> dict:
    """Ingest every PDF under DATA_DIR (idempotent — upserts by candidate id)."""
    s = get_settings()
    store = get_rag_client().store
    pdfs = [
        os.path.join(s.DATA_DIR, f)
        for f in sorted(os.listdir(s.DATA_DIR))
        if f.lower().endswith(".pdf")
    ]
    for pdf in pdfs:
        ingest_pdf(pdf, store)
    return {"ingested": len(pdfs), "candidates": store.count() // 4}
